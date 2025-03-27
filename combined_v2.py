import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import re
from ortools.linear_solver import pywraplp
from datetime import datetime, timedelta, date
from collections import defaultdict
from ast import literal_eval
import random
import numpy as np
import plotly.express as px
import calendar

# Initial NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# First Application Setup (Contextual RAG Q&A)
load_dotenv()
ADMIN_CREDENTIALS = {"admin_id": "admin", "password": "admin123"}
conn = sqlite3.connect('files.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS files
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT UNIQUE,
              content TEXT,
              uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Second Application Setup (Shipping Resource Allocator)
class ShippingResourceAllocator:
    def __init__(self):
        self.employees = []
        self.vessels = []
        self.voyages = []
        self.skill_requirements = {}
        
    def load_data(self, employees_data, vessels_data, voyages_data):
        try:
            self.employees = employees_data.to_dict('records')
            for employee in self.employees:
                employee['skills'] = self._parse_skills(employee.get('skills', {}))
            self.vessels = vessels_data.to_dict('records')
            self.voyages = voyages_data.copy()
            self.voyages['start_date'] = pd.to_datetime(self.voyages['start_date'])
            self.voyages['end_date'] = pd.to_datetime(self.voyages['end_date'])
            self.voyages = self.voyages.to_dict('records')
            if not all(['employee_id' in e for e in self.employees]):
                raise ValueError("Employee data missing employee_id")
            if not all(['vessel_id' in v for v in self.vessels]):
                raise ValueError("Vessel data missing vessel_id")
            if not all(['voyage_id' in v for v in self.voyages]):
                raise ValueError("Voyage data missing voyage_id")
            self._process_skill_requirements()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _parse_skills(self, skills_data):
        if isinstance(skills_data, dict):
            return skills_data
        if isinstance(skills_data, str):
            try:
                return json.loads(skills_data.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    return literal_eval(skills_data)
                except:
                    return {}
        return {}
    
    def _process_skill_requirements(self):
        self.skill_requirements = {
            'Container Ship': {
                'min_crew': 5,
                'required_skills': {
                    'navigation': 3,
                    'cargo_handling': 2,
                    'safety_training': 3
                }
            },
            'Tanker': {
                'min_crew': 6,
                'required_skills': {
                    'hazardous_materials': 2,
                    'navigation': 3,
                    'safety_training': 3
                }
            }
        }
    
    def _get_employee_skill_level(self, employee, skill):
        return float(employee['skills'].get(skill, 0))
    
    def optimize_allocation(self, start_date, end_date):
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            elif isinstance(start_date, date):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            elif isinstance(end_date, date):
                end_date = pd.to_datetime(end_date)
            relevant_voyages = [
                v for v in self.voyages 
                if not (v['end_date'] < start_date or v['start_date'] > end_date)
            ]
            if not relevant_voyages:
                return {"status": "No voyages in specified period"}
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                return {"status": "Failed to create solver"}
            assignments = {}
            for e in self.employees:
                for v in relevant_voyages:
                    assignments[(e['employee_id'], v['voyage_id'])] = solver.IntVar(
                        0, 1, f"x_{e['employee_id']}_{v['voyage_id']}")
            objective = solver.Objective()
            for (e_id, v_id), var in assignments.items():
                employee = next((e for e in self.employees if e['employee_id'] == e_id), None)
                voyage = next((v for v in relevant_voyages if v['voyage_id'] == v_id), None)
                if not employee or not voyage:
                    continue
                vessel = next((v for v in self.vessels if v['vessel_id'] == voyage['vessel_id']), None)
                if not vessel:
                    continue
                skill_score = self._calculate_skill_match(employee, vessel['type'])
                cost = employee.get('daily_cost', 0) * (voyage['end_date'] - voyage['start_date']).days
                objective.SetCoefficient(var, skill_score - 0.1 * cost)
            objective.SetMaximization()
            self._add_availability_constraints(solver, assignments, relevant_voyages, start_date, end_date)
            self._add_crew_size_constraints(solver, assignments, relevant_voyages)
            self._add_skill_constraints(solver, assignments, relevant_voyages)
            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                return self._prepare_results(assignments, relevant_voyages)
            return {"status": f"No optimal solution found (status: {status})"}
        except Exception as e:
            return {"status": f"Optimization failed: {str(e)}"}
    
    def _add_availability_constraints(self, solver, assignments, voyages, start_date, end_date):
        if isinstance(start_date, date):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, date):
            end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start_date, end_date)
        for e in self.employees:
            for day in date_range:
                constraint = solver.Constraint(0, 1)
                for v in voyages:
                    if v['start_date'] <= day <= v['end_date']:
                        var = assignments.get((e['employee_id'], v['voyage_id']))
                        if var:
                            constraint.SetCoefficient(var, 1)
    
    def _add_crew_size_constraints(self, solver, assignments, voyages):
        for v in voyages:
            vessel = next((vsl for vsl in self.vessels if vsl['vessel_id'] == v['vessel_id']), None)
            if not vessel:
                continue
            vessel_type = vessel.get('type')
            if not vessel_type:
                continue
            min_crew = self.skill_requirements.get(vessel_type, {}).get('min_crew', 0)
            if min_crew <= 0:
                continue
            constraint = solver.Constraint(min_crew, solver.infinity())
            for e in self.employees:
                var = assignments.get((e['employee_id'], v['voyage_id']))
                if var:
                    constraint.SetCoefficient(var, 1)
    
    def _add_skill_constraints(self, solver, assignments, voyages):
        for v in voyages:
            vessel = next((vsl for vsl in self.vessels if vsl['vessel_id'] == v['vessel_id']), None)
            if not vessel:
                continue
            vessel_type = vessel.get('type')
            if not vessel_type:
                continue
            req_skills = self.skill_requirements.get(vessel_type, {}).get('required_skills', {})
            for skill, min_level in req_skills.items():
                constraint = solver.Constraint(min_level, solver.infinity())
                for e in self.employees:
                    skill_level = self._get_employee_skill_level(e, skill)
                    if skill_level >= 1:
                        var = assignments.get((e['employee_id'], v['voyage_id']))
                        if var:
                            constraint.SetCoefficient(var, skill_level)
    
    def _calculate_skill_match(self, employee, vessel_type):
        if vessel_type not in self.skill_requirements:
            return 0
        total_score = 0
        for skill, min_level in self.skill_requirements[vessel_type]['required_skills'].items():
            skill_level = self._get_employee_skill_level(employee, skill)
            if skill_level >= 1:
                total_score += min(skill_level, min_level) * 1.0
                if skill_level > min_level:
                    total_score += (skill_level - min_level) * 0.5
        return total_score
    
    def _prepare_results(self, assignments, voyages):
        allocation = defaultdict(list)
        for (e_id, v_id), var in assignments.items():
            if var.solution_value() > 0.5:
                allocation[v_id].append(e_id)
        return {
            "status": "OPTIMAL",
            "total_assignments": sum(var.solution_value() for var in assignments.values()),
            "allocations": dict(allocation)
        }
    
    def generate_report(self, allocation_result):
        if not isinstance(allocation_result, dict):
            return {"status": "Invalid allocation result format"}
        if allocation_result.get('status') != 'OPTIMAL':
            return {"status": f"No optimal allocation: {allocation_result.get('status', 'Unknown error')}"}
        if not allocation_result.get('allocations'):
            return {"status": "No allocations found in results"}
        report = []
        for v_id, e_ids in allocation_result['allocations'].items():
            voyage = next((v for v in self.voyages if v['voyage_id'] == v_id), None)
            if not voyage:
                continue
            vessel = next((v for v in self.vessels if v['vessel_id'] == voyage['vessel_id']), None)
            if not vessel:
                continue
            voyage_report = {
                'voyage_id': v_id,
                'vessel': vessel.get('name', 'Unknown'),
                'type': vessel.get('type', 'Unknown'),
                'route': voyage.get('route', 'Unknown'),
                'dates': f"{voyage.get('start_date', 'Unknown').strftime('%Y-%m-%d')} to {voyage.get('end_date', 'Unknown').strftime('%Y-%m-%d')}",
                'crew_count': len(e_ids),
                'crew': []
            }
            for e_id in e_ids:
                employee = next((e for e in self.employees if e['employee_id'] == e_id), None)
                if employee:
                    voyage_report['crew'].append({
                        'id': e_id,
                        'name': employee.get('name', 'Unknown'),
                        'position': employee.get('position', 'Unknown'),
                        'skills': employee.get('skills', {})
                    })
            report.append(voyage_report)
        return report if report else {"status": "No valid voyage allocations to report"}

# Third Application Setup (Ship Maintenance System)
def generate_random_maintenance_data(num_records=100):
    ships = ['Titanic', 'Queen Mary', 'Black Pearl', 'Flying Dutchman', 'SS Minnow']
    components = ['Engine', 'Propeller', 'Navigation System', 'Hull', 'Electrical System', 
                  'Fuel System', 'Cooling System', 'Deck Equipment', 'Safety Equipment']
    maintenance_types = ['Preventive', 'Corrective', 'Predictive', 'Condition-based']
    statuses = ['Completed', 'Pending', 'Overdue', 'Cancelled']
    data = []
    for _ in range(num_records):
        ship = random.choice(ships)
        component = random.choice(components)
        maintenance_type = random.choice(maintenance_types)
        status = random.choice(statuses)
        last_date = datetime.now() - timedelta(days=random.randint(1, 365))
        next_date = last_date + timedelta(days=random.randint(30, 365))
        cost = round(random.uniform(100, 10000), 2)
        data.append({
            'Ship': ship,
            'Component': component,
            'Maintenance Type': maintenance_type,
            'Last Maintenance Date': last_date,
            'Next Maintenance Date': next_date,
            'Status': status,
            'Cost ($)': cost,
            'Hours Spent': random.randint(1, 48)
        })
    return pd.DataFrame(data)

def generate_sensor_data(num_records=500):
    components = ['Engine', 'Propeller', 'Cooling System', 'Electrical System']
    parameters = {
        'Engine': ['Temperature', 'Pressure', 'Vibration', 'Oil Level'],
        'Propeller': ['RPM', 'Vibration', 'Lubrication', 'Wear'],
        'Cooling System': ['Temperature', 'Flow Rate', 'Pressure', 'Coolant Level'],
        'Electrical System': ['Voltage', 'Current', 'Resistance', 'Temperature']
    }
    data = []
    for _ in range(num_records):
        component = random.choice(components)
        parameter = random.choice(parameters[component])
        if parameter == 'Temperature':
            value = round(random.uniform(50, 120), 2)
            threshold = 100
        elif parameter == 'Pressure':
            value = round(random.uniform(1, 10), 2)
            threshold = 8
        elif parameter == 'Vibration':
            value = round(random.uniform(0.1, 5), 2)
            threshold = 3.5
        elif parameter == 'Oil Level':
            value = round(random.uniform(1, 10), 2)
            threshold = 2
        elif parameter == 'RPM':
            value = random.randint(100, 1000)
            threshold = 900
        elif parameter == 'Lubrication':
            value = round(random.uniform(0, 100), 2)
            threshold = 30
        elif parameter == 'Wear':
            value = round(random.uniform(0, 100), 2)
            threshold = 80
        elif parameter == 'Flow Rate':
            value = round(random.uniform(1, 20), 2)
            threshold = 15
        elif parameter == 'Coolant Level':
            value = round(random.uniform(1, 10), 2)
            threshold = 2
        elif parameter == 'Voltage':
            value = round(random.uniform(200, 500), 2)
            threshold = 480
        elif parameter == 'Current':
            value = round(random.uniform(1, 50), 2)
            threshold = 45
        elif parameter == 'Resistance':
            value = round(random.uniform(0.1, 100), 2)
            threshold = 90
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 10080))
        alert = value > threshold
        data.append({
            'Timestamp': timestamp,
            'Component': component,
            'Parameter': parameter,
            'Value': value,
            'Threshold': threshold,
            'Alert': alert
        })
    return pd.DataFrame(data)

# Application Functions
def app1():
    def extract_text_from_file(file):
        text = ""
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file.type == "application/json":
            json_data = json.load(file)
            text = json.dumps(json_data, indent=4)
        elif file.type == "text/markdown":
            text = file.getvalue().decode("utf-8")
        elif file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_csv(file) if file.type == "text/csv" else pd.read_excel(file)
            text = df.to_string()
        return text

    def chunk_text(text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length and end - start == chunk_size:
                last_period = max(text.rfind('.', start, end), text.rfind('\n', start, end))
                if last_period > start + chunk_size // 2: 
                    end = last_period + 1
            chunks.append(text[start:end])
            start = end - overlap if end < text_length else text_length
        return chunks

    def preprocess_text(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return word_tokenize(text)

    def store_file_in_db(filename, content):
        c.execute("SELECT COUNT(*) FROM files WHERE filename = ?", (filename,))
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO files (filename, content) VALUES (?, ?)", (filename, content))
            conn.commit()
            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                embedding = embedding_model.encode(chunk).tolist()
                collection.add(documents=[chunk], embeddings=[embedding], ids=[chunk_id], metadatas=[{"filename": filename, "chunk_index": i}])
            st.success(f"'{filename}' uploaded successfully!")
        else:
            st.warning(f"'{filename}' is already uploaded.")

    def retrieve_relevant_context(query, top_k=10):
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k*2)
        if not results["documents"][0]:
            return ""
        retrieved_docs = results["documents"][0]
        doc_ids = results["ids"][0]
        tokenized_query = preprocess_text(query)
        tokenized_docs = [preprocess_text(doc) for doc in retrieved_docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query)
        vector_scores = [1.0 - results["distances"][0][i] for i in range(len(retrieved_docs))]
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        max_vector = max(vector_scores) if vector_scores else 1.0
        normalized_bm25 = [score/max_bm25 for score in bm25_scores]
        normalized_vector = [score/max_vector for score in vector_scores]
        alpha = 0.5
        combined_scores = [(alpha * v_score + (1-alpha) * bm_score)
                          for v_score, bm_score in zip(normalized_vector, normalized_bm25)]
        doc_score_pairs = list(zip(retrieved_docs, doc_ids, combined_scores))
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[2], reverse=True)[:top_k]
        context_chunks = []
        for doc, doc_id, score in ranked_docs:
            metadata = f"Source: {doc_id.split('_chunk_')[0]} (Score: {score:.3f})"
            context_chunks.append(f"{metadata}\n{doc}")
        return "\n\n" + "-"*50 + "\n\n".join(context_chunks)

    def azure_openai_query(question, context, api_key, endpoint):
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )
        system_prompt = """
            You are an AI assistant tasked with performing comprehensive extraction of information from ALL provided context chunks.
            When responding to a user's question, adhere strictly to the following guidelines:
            - Review EVERY context chunk provided thoroughly, ensuring you cover ALL occurrences of relevant information.
            - Extract and list EVERY relevant piece of information explicitly and separately, even if the same or similar information appears multiple times or across different chunks.
            - Do NOT stop after partial matches or the initial findings; CONTINUE reviewing ALL chunks until no additional relevant information remains.
            - Clearly format your responses in a structured manner (e.g., bullet points or numbered lists) for readability.
            Provide explicit source citations indicating:
            - The exact document filename.
            - The specific chunk or page number from which each piece of information was extracted.
            Your goal is complete accuracy and exhaustive retrieval—no relevant data should be omitted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()

    st.title("Contextual RAG Q&A Web Application")
    tab1, tab2, tab3 = st.tabs(["Admin", "Credentials", "Chat"])
    with tab1:
        st.subheader("Admin Panel")
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False
        if not st.session_state.admin_logged_in:
            admin_id = st.text_input("Admin ID")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if admin_id == ADMIN_CREDENTIALS["admin_id"] and password == ADMIN_CREDENTIALS["password"]:
                    st.session_state.admin_logged_in = True
                    st.success("Logged in as Admin")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        if st.session_state.admin_logged_in:
            uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt", "docx", "json", "md", "csv", "xlsx"])
            if uploaded_files:
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    store_file_in_db(file.name, text)
            st.subheader("Uploaded Files")
            c.execute("SELECT filename FROM files")
            files = c.fetchall()
            if files:
                df = pd.DataFrame(files, columns=["Filename"])
                for filename in df["Filename"]:
                    col1, col2 = st.columns([4, 1])
                    col1.text(filename)
                    if col2.button("Delete", key=filename):
                        c.execute("DELETE FROM files WHERE filename = ?", (filename,))
                        conn.commit()
                        try:
                            file_chunks = collection.get(where={"filename": filename})
                            if file_chunks and "ids" in file_chunks and file_chunks["ids"]:
                                collection.delete(ids=file_chunks["ids"])
                        except Exception as e:
                            st.error(f"Error deleting from ChromaDB: {e}")
                        st.rerun()
            else:
                st.info("No files uploaded yet.")
            if st.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()
    with tab2:
        st.subheader("Azure OpenAI Credentials")
        st.session_state.api_key = st.text_input("Enter Azure OpenAI API Key", type="password")
        st.session_state.endpoint = st.text_input("Enter Azure OpenAI Endpoint")
    with tab3:
        st.subheader("Chat Interface")
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])
        if 'show_context' not in st.session_state:
            st.session_state.show_context = False
        show_context = st.sidebar.checkbox("Show retrieved context", value=st.session_state.show_context)
        st.session_state.show_context = show_context
        question = st.chat_input("Ask me anything...")
        if question:
            st.chat_message("user").markdown(question)
            with st.status("Searching documents...", expanded=True) as status:
                context = retrieve_relevant_context(question)
                if not context:
                    status.update(label="No relevant documents found", state="error")
                    answer = "I couldn't find any relevant information in the knowledge base. Please try a different question or upload more documents."
                else:
                    if st.session_state.show_context:
                        st.sidebar.markdown("### Retrieved Context")
                        st.sidebar.markdown(context)
                    status.update(label="Generating answer...", state="running")
                    answer = azure_openai_query(question, context, st.session_state.api_key, st.session_state.endpoint)
                    status.update(label="Answer generated!", state="complete")
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})

def app2():
    def generate_random_data(num_employees=10, num_vessels=3, num_voyages=3):
        skill_pool = {
            'navigation': [1, 2, 3, 4, 5],
            'cargo_handling': [1, 2, 3, 4],
            'safety_training': [2, 3, 4, 5],
            'engine_maintenance': [1, 2, 3, 4, 5],
            'hazardous_materials': [1, 2, 3, 4, 5],
            'medical_training': [1, 2, 3],
            'communication': [2, 3, 4, 5]
        }
        positions = ['Captain', 'First Mate', 'Chief Engineer', 'Deck Officer', 
                    'Engineer', 'Deckhand', 'Cook', 'Medical Officer']
        employees = []
        for i in range(1, num_employees + 1):
            num_skills = random.randint(2, 5)
            skills = {}
            for skill in random.sample(list(skill_pool.keys()), num_skills):
                skills[skill] = random.choice(skill_pool[skill])
            employees.append({
                'employee_id': 100 + i,
                'name': f"Employee {i}",
                'position': random.choice(positions),
                'skills': skills,
                'daily_cost': random.randint(200, 400)
            })
        vessel_types = ['Container Ship', 'Tanker', 'Bulk Carrier', 'Ro-Ro']
        vessels = []
        for i in range(1, num_vessels + 1):
            vessels.append({
                'vessel_id': 200 + i,
                'name': f"Vessel {i}",
                'type': random.choice(vessel_types),
                'capacity': random.randint(5000, 50000)
            })
        routes = ['Shanghai to Los Angeles', 'Singapore to Rotterdam', 
                  'Houston to Hamburg', 'Dubai to Mumbai', 'Sydney to Auckland']
        voyages = []
        base_date = datetime(2025, 3, 26)
        for i in range(1, num_voyages + 1):
            start_date = base_date + timedelta(days=random.randint(0, 7))
            duration = random.randint(7, 21)
            end_date = start_date + timedelta(days=duration)
            voyages.append({
                'voyage_id': 300 + i,
                'vessel_id': random.choice([v['vessel_id'] for v in vessels]),
                'route': random.choice(routes),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
        return (
            pd.DataFrame(employees),
            pd.DataFrame(vessels),
            pd.DataFrame(voyages)
        )

    allocator = ShippingResourceAllocator()
    st.title("Shipping Resource Allocation System")
    st.sidebar.header("Configuration")
    use_random_data = st.sidebar.checkbox("Use Random Data", value=True)
    random.seed(st.sidebar.number_input("Random Seed", value=42))
    if use_random_data:
        num_employees = st.sidebar.slider("Number of Employees", 5, 50, 15)
        num_vessels = st.sidebar.slider("Number of Vessels", 1, 10, 3)
        num_voyages = st.sidebar.slider("Number of Voyages", 1, 10, 3)
        employees_df, vessels_df, voyages_df = generate_random_data(
            num_employees, num_vessels, num_voyages)
    else:
        employees_file = st.sidebar.file_uploader("Upload Employees CSV", type="csv")
        vessels_file = st.sidebar.file_uploader("Upload Vessels CSV", type="csv")
        voyages_file = st.sidebar.file_uploader("Upload Voyages CSV", type="csv")
        if not (employees_file and vessels_file and voyages_file):
            st.warning("Please upload all required files")
            return
        employees_df = pd.read_csv(employees_file)
        vessels_df = pd.read_csv(vessels_file)
        voyages_df = pd.read_csv(voyages_file)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", date(2025, 3, 26))
    with col2:
        end_date = st.date_input("End Date", date(2025, 4, 10))
    st.subheader("Data Preview")
    tab1, tab2, tab3 = st.tabs(["Employees", "Vessels", "Voyages"])
    with tab1:
        st.dataframe(employees_df)
    with tab2:
        st.dataframe(vessels_df)
    with tab3:
        st.dataframe(voyages_df)
    if st.button("Run Optimization"):
        with st.spinner("Loading data and optimizing..."):
            if not allocator.load_data(employees_df, vessels_df, voyages_df):
                st.error("Failed to load data")
                return
            st.success("Data loaded successfully:")
            st.write(f"- Employees: {len(allocator.employees)}")
            st.write(f"- Vessels: {len(allocator.vessels)}")
            st.write(f"- Voyages: {len(allocator.voyages)}")
            result = allocator.optimize_allocation(start_date, end_date)
            st.subheader("Optimization Results")
            st.write(f"Status: {result.get('status')}")
            report = allocator.generate_report(result)
            if isinstance(report, list):
                st.success(f"Found {len(report)} voyage allocations:")
                for voyage in report:
                    with st.expander(f"Voyage {voyage['voyage_id']}: {voyage['vessel']} ({voyage['type']})"):
                        st.write(f"**Route:** {voyage['route']}")
                        st.write(f"**Dates:** {voyage['dates']}")
                        st.write(f"**Crew Members ({voyage['crew_count']}):**")
                        crew_df = pd.DataFrame(voyage['crew'])
                        st.dataframe(crew_df)
                        skills_data = []
                        for crew in voyage['crew']:
                            for skill, level in crew['skills'].items():
                                skills_data.append({
                                    'Name': crew['name'],
                                    'Skill': skill,
                                    'Level': level
                                })
                        if skills_data:
                            skills_df = pd.DataFrame(skills_data)
                            st.subheader("Crew Skills")
                            st.bar_chart(skills_df.pivot(index='Name', columns='Skill', values='Level'))
            else:
                st.warning(f"Report: {report.get('status')}")

def app3():
    if 'maintenance_data' not in st.session_state:
        st.session_state.maintenance_data = generate_random_maintenance_data(200)
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = generate_sensor_data(1000)
    st.sidebar.header("Filters")
    selected_ship = st.sidebar.selectbox("Select Ship", ['All'] + list(st.session_state.maintenance_data['Ship'].unique()))
    selected_component = st.sidebar.selectbox("Select Component", ['All'] + list(st.session_state.maintenance_data['Component'].unique()))
    selected_status = st.sidebar.selectbox("Select Status", ['All'] + list(st.session_state.maintenance_data['Status'].unique()))
    selected_type = st.sidebar.selectbox("Select Maintenance Type", ['All'] + list(st.session_state.maintenance_data['Maintenance Type'].unique()))
    filtered_data = st.session_state.maintenance_data.copy()
    if selected_ship != 'All':
        filtered_data = filtered_data[filtered_data['Ship'] == selected_ship]
    if selected_component != 'All':
        filtered_data = filtered_data[filtered_data['Component'] == selected_component]
    if selected_status != 'All':
        filtered_data = filtered_data[filtered_data['Status'] == selected_status]
    if selected_type != 'All':
        filtered_data = filtered_data[filtered_data['Maintenance Type'] == selected_type]
    st.title("🚢 Ship Maintenance Management System")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_maintenance = len(filtered_data)
        st.metric("Total Maintenance Records", total_maintenance)
    with col2:
        preventive_count = len(filtered_data[filtered_data['Maintenance Type'] == 'Preventive'])
        st.metric("Preventive Maintenance", preventive_count)
    with col3:
        overdue_count = len(filtered_data[filtered_data['Status'] == 'Overdue'])
        st.metric("Overdue Maintenance", overdue_count)
    with col4:
        total_cost = filtered_data['Cost ($)'].sum()
        st.metric("Total Cost ($)", f"{total_cost:,.2f}")
    tab1, tab2, tab3, tab4 = st.tabs(["Maintenance Records", "Preventive Schedule", "Predictive Analytics", "Add New Record"])
    with tab1:
        st.subheader("Maintenance Records")
        st.dataframe(filtered_data.sort_values('Next Maintenance Date', ascending=False), 
                    use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Maintenance by Type")
            type_counts = filtered_data['Maintenance Type'].value_counts()
            fig = px.pie(type_counts, values=type_counts.values, names=type_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Cost Distribution by Component")
            cost_by_component = filtered_data.groupby('Component')['Cost ($)'].sum().reset_index()
            fig = px.bar(cost_by_component, x='Component', y='Cost ($)', color='Component')
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Preventive Maintenance Schedule")
        months = [calendar.month_name[i] for i in range(1, 13)]
        selected_month = st.selectbox("Select Month", months, index=datetime.now().month-1)
        month_num = months.index(selected_month) + 1
        current_year = datetime.now().year
        preventive_data = filtered_data[filtered_data['Maintenance Type'] == 'Preventive']
        monthly_schedule = preventive_data[
            (preventive_data['Next Maintenance Date'].dt.month == month_num) & 
            (preventive_data['Next Maintenance Date'].dt.year == current_year)
        ]
        if not monthly_schedule.empty:
            st.dataframe(monthly_schedule.sort_values('Next Maintenance Date'), 
                        use_container_width=True, height=400)
            fig = px.timeline(
                monthly_schedule, 
                x_start="Last Maintenance Date", 
                x_end="Next Maintenance Date", 
                y="Component",
                color="Ship",
                title="Maintenance Schedule Timeline"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No preventive maintenance scheduled for the selected month.")
    with tab3:
        st.subheader("Predictive Maintenance Analytics")
        alert_data = st.session_state.sensor_data[st.session_state.sensor_data['Alert'] == True]
        if not alert_data.empty:
            st.warning(f"⚠️ {len(alert_data)} active alerts detected!")
            st.dataframe(alert_data.sort_values('Timestamp', ascending=False), 
                        use_container_width=True, height=300)
            selected_component_alert = st.selectbox(
                "Select Component for Analysis", 
                alert_data['Component'].unique()
            )
            component_data = st.session_state.sensor_data[
                st.session_state.sensor_data['Component'] == selected_component_alert
            ]
            fig = px.line(
                component_data, 
                x='Timestamp', 
                y='Value', 
                color='Parameter',
                title=f"Sensor Data for {selected_component_alert}",
                markers=True
            )
            for parameter in component_data['Parameter'].unique():
                threshold = component_data[component_data['Parameter'] == parameter]['Threshold'].iloc[0]
                fig.add_hline(
                    y=threshold, 
                    line_dash="dot",
                    annotation_text=f"{parameter} Threshold",
                    annotation_position="bottom right",
                    line_color="red"
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No active alerts detected. All systems operating within normal parameters.")
    with tab4:
        st.subheader("Add New Maintenance Record")
        with st.form("maintenance_form"):
            col1, col2 = st.columns(2)
            with col1:
                ship = st.selectbox("Ship", st.session_state.maintenance_data['Ship'].unique())
                component = st.selectbox("Component", st.session_state.maintenance_data['Component'].unique())
                maintenance_type = st.selectbox("Maintenance Type", st.session_state.maintenance_data['Maintenance Type'].unique())
                status = st.selectbox("Status", st.session_state.maintenance_data['Status'].unique())
            with col2:
                last_maintenance_date = st.date_input("Last Maintenance Date", datetime.now())
                next_maintenance_date = st.date_input("Next Maintenance Date", datetime.now() + timedelta(days=30))
                cost = st.number_input("Cost ($)", min_value=0.0, value=1000.0, step=100.0)
                hours_spent = st.number_input("Hours Spent", min_value=1, value=8, step=1)
            submitted = st.form_submit_button("Add Record")
            if submitted:
                new_record = {
                    'Ship': ship,
                    'Component': component,
                    'Maintenance Type': maintenance_type,
                    'Last Maintenance Date': last_maintenance_date,
                    'Next Maintenance Date': next_maintenance_date,
                    'Status': status,
                    'Cost ($)': cost,
                    'Hours Spent': hours_spent
                }
                new_df = pd.DataFrame([new_record])
                st.session_state.maintenance_data = pd.concat([st.session_state.maintenance_data, new_df], ignore_index=True)
                st.success("Maintenance record added successfully!")
                st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)

# Main Application
def main():
    st.set_page_config(page_title="Shipping Management Suite", layout="wide")
    st.title("Shipping Management Suite")

    # Initialize selected_app in session state if not present
    if 'selected_app' not in st.session_state:
        st.session_state.selected_app = "Contextual RAG Q&A"

    # Custom CSS for button styling
    st.markdown("""
        <style>
        .app-button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            text-align: center;
        }
        .app-button-unselected {
            background-color: #f0f0f0;
            color: #000000;
        }
        .app-button-selected {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create three columns for the buttons
    col1, col2, col3 = st.columns(3)

    # Button for Contextual RAG Q&A
    with col1:
        if st.button("Contextual RAG Q&A", key="app1_btn"):
            st.session_state.selected_app = "Contextual RAG Q&A"
        button_style = "app-button-selected" if st.session_state.selected_app == "Contextual RAG Q&A" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Contextual RAG Q&A</div>', unsafe_allow_html=True)

    # Button for Shipping Resource Allocator
    with col2:
        if st.button("Shipping Resource Allocator", key="app2_btn"):
            st.session_state.selected_app = "Shipping Resource Allocator"
        button_style = "app-button-selected" if st.session_state.selected_app == "Shipping Resource Allocator" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Shipping Resource Allocator</div>', unsafe_allow_html=True)

    # Button for Ship Maintenance System
    with col3:
        if st.button("Ship Maintenance System", key="app3_btn"):
            st.session_state.selected_app = "Ship Maintenance System"
        button_style = "app-button-selected" if st.session_state.selected_app == "Ship Maintenance System" else "app-button-unselected"
        st.markdown(f'<div class="{button_style} app-button">Ship Maintenance System</div>', unsafe_allow_html=True)

    # Display selected application
    st.markdown("---")
   
    if st.session_state.selected_app == "Shipping Resource Allocator":
        app2()
    elif st.session_state.selected_app == "Ship Maintenance System":
        app3()
    elif st.session_state.selected_app == "Contextual RAG Q&A":
         app1()

if __name__ == "__main__":
    main()