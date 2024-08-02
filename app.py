import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from neo4j import GraphDatabase
from py2neo import Graph
from pyvis.network import Network
import matplotlib.pyplot as plt
import io
from PIL import Image
import openai
import os
from dotenv import load_dotenv
from openai.error import OpenAIError
import streamlit.components.v1 as components
from py2neo import Graph, Node, Relationship
import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
password = os.getenv("NEO4J_PASSWORD")
user = os.getenv("NEO4J_USER")
uri = os.getenv("NEO4J_URI")
data_from_db = []
# if not openai.api_key:
#     st.error("OpenAI API key not found. Please set the API key in your environment variables.")
# else:
#     st.write(f"OpenAI API key loaded: {openai.api_key[:5]}...{openai.api_key[-5:]}")

def create_neo4j_driver(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver
    except Exception as e:
        print(f"Failed to create Neo4j driver: {e}")

# # Function to create a Neo4j driver
# def create_neo4j_driver(uri, user, password):
#     return GraphDatabase.driver(uri, auth=(user, password))

# Function to save PDF info to Neo4j
def save_pdf_info(driver, filename):
    with driver.session() as session:
        session.write_transaction(create_pdf_node, filename)

# Function to create a PDF node in Neo4j
def create_pdf_node(tx, filename):
    tx.run("CREATE (p:PDF {name: $filename})", filename=filename)

def process_pdf_text(text):
    # Regular expressions to match the required patterns
    report_pattern = re.compile(r'(\d+)\.\s*([\w\s\-]+):\s*([\w\s\-\'\&\,\.]+)')
    regulator_pattern = re.compile(r'(Board of Governors of Federal Reserve System|Office of Comptroller of the Currency|Federal Deposit Insurance Corporation)')
    
    # Lists to store the extracted data
    reports = []
    regulators = []
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Process each line
    for line in lines:
        report_match = report_pattern.match(line)
        if report_match:
            report_id = report_match.group(1).strip()
            report_code = report_match.group(2).strip()
            report_name = report_match.group(3).strip()
            reports.append([report_id, report_code, report_name])
        
        regulator_match = regulator_pattern.search(line)
        if regulator_match:
            regulators.append(regulator_match.group(1).strip())
    
    reports_df = pd.DataFrame(reports, columns=['Report_ID', 'Report_Code', 'Report_Name'])
    regulators_df = pd.DataFrame(regulators, columns=['Regulator'])
    
    return reports_df, regulators_df

def save_nodes_and_relationships(graph, reports_df, regulators_df):
    for index, row in reports_df.iterrows():
        id_node = Node("ReportID", name=row['Report_ID'])
        code_node = Node("ReportCode", name=row['Report_Code'])
        name_node = Node("ReportName", name=row['Report_Name'])
        
        # Merge nodes in the graph
        graph.merge(id_node, "ReportID", "name")
        graph.merge(code_node, "ReportCode", "name")
        graph.merge(name_node, "ReportName", "name")
        
        # Create relationships between nodes
        id_to_code_relationship = Relationship(id_node, "HAS_CODE", code_node)
        code_to_name_relationship = Relationship(code_node, "HAS_NAME", name_node)
        
        # Merge relationships in the graph
        graph.merge(id_to_code_relationship)
        graph.merge(code_to_name_relationship)
    
    for index, row in regulators_df.iterrows():
        regulator_node = Node("Regulator", name=row['Regulator'])
        graph.merge(regulator_node, "Regulator", "name")
    
    print("Data imported into Neo4j graph database successfully.")


# # Function to query OpenAI
# def query_openai(prompt, context=""):
#     full_prompt = f"{context}\n\n{prompt}"
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": full_prompt}
#             ],
#             max_tokens=150
#         )
#         return response.choices[0].message['content'].strip()
#     except openai.OpenAIError as e:
#         return f"An error occurred: {str(e)}"
    
def query_openai(prompt, context=""):
    full_prompt = f"{context}\n\n{prompt}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except openai.OpenAIError as e:
        return f"An error occurred: {str(e)}"

def query_neo4j(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return result.data()

    
def fetch_data_based_on_prompt(driver, prompt):
    cypher_query = generate_cypher_query(prompt)
    print(cypher_query)
    if not cypher_query:
        print("not cypher")
        cypher_query = query_openai(prompt, context="Translate the following English prompt into a Cypher query")
    global data_from_db
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            data_from_db = result.data()
            print(f"Data from Neo4j: {data_from_db}")
            return data_from_db
    except Exception as e:
        print(f"Failed to execute query: {e}")
        return []
    
# def fetch_graph_data(uri, user, password):
#     graph = Graph(uri, auth=(user, password))
#     return graph.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100").data()

# def visualize_graph(data):
#     from pyvis.network import Network
    
#     net = Network(notebook=True)
#     nodes = set()
#     for record in data:
#         start_node = record['n']
#         end_node = record['m']
#         if start_node.identity not in nodes:
#             net.add_node(start_node.identity, label=start_node['name'])
#             nodes.add(start_node.identity)
#         if end_node.identity not in nodes:
#             net.add_node(end_node.identity, label=end_node['name'])
#             nodes.add(end_node.identity)
#         net.add_edge(start_node.identity, end_node.identity, title=record['r'].type)
    
#     # Save visualization to an HTML file
#     html_path = 'graph.html'
#     net.save_graph(html_path)
#     return html_path

def generate_cypher_query(prompt):
    doc = nlp(prompt)
    entities = [ent.text for ent in doc.ents]
    # Keywords and their corresponding Cypher queries
    if "explain" in prompt.lower() or "what is" in prompt.lower():
        # Extract information about a specific report or regulator
        if entities:
            entity_name = entities[0]
            # Use the entity name directly to search in the ReportName or Regulator
            return f"MATCH (n:ReportCode {{name: '{entity_name}'}}) RETURN n"

    if "list" in prompt.lower() or "show" in prompt.lower():
        # List all reports or regulators
        if "reports" in prompt.lower():
            return "MATCH (r:ReportName) RETURN r.name AS ReportName"
        elif "regulators" in prompt.lower():
            return "MATCH (r:Regulator) RETURN r.name AS Regulator"

    if "relationship between" in prompt.lower():
        # Retrieve the relationship between two entities
        if len(entities) >= 2:
            entity1, entity2 = entities[:2]
            return f"""
            MATCH (a {{name: '{entity1}'}})-[r]->(b {{name: '{entity2}'}})
            RETURN a, r, b
            """

    # General queries with keywords like "find" and "details of"
    if "find" in prompt.lower() or "details of" in prompt.lower():
        if entities:
            entity_name = entities[0]
            return f"""
            MATCH (r:ReportCode {{name: '{entity_name}'}})
            OPTIONAL MATCH (r)-[:HAS_CODE]->(c:ReportCode)
            OPTIONAL MATCH (c)-[:HAS_NAME]->(n:ReportName)
            RETURN r, c, n
            """

    # Fallback case if no specific rule matched
    return None

def evaluate_response(response, context, question):
    evaluation_prompt = f"Context: {context}\n\nQuestion: {question}\n\nResponse: {response}\n\nEvaluate the response based on correctness, coherence, and relevance. Provide a score out of 10 and a brief explanation."
    try:
        evaluation = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=150
        )
        return evaluation.choices[0].message['content'].strip()
    except openai.OpenAIError as e:
        return f"An error occurred during evaluation: {str(e)}"

# Main function for the Streamlit app
def main():

    global uri, user, password, data_from_db
    st.title("Streamlit App with Neo4j and OpenAI Integration")

    st.sidebar.title("Navigation")
    options = ["Home", "PDF Upload with Neo4j Integration", "Query OpenAI"]  # , "View Knowledge Graph"
    choice = st.sidebar.selectbox("Select a page", options)

    if choice == "Home":
        st.header("Welcome to the Streamlit App")
        st.write("This app integrates Neo4j and OpenAI to handle queries and provide insights.")

    elif choice == "PDF Upload with Neo4j Integration":
        st.header("PDF Upload with Neo4j Integration")

        # Input fields for Neo4j connection
        st.sidebar.header("Neo4j Connection")
        uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
        user = st.sidebar.text_input("User", "neo4j")
        password = st.sidebar.text_input("Password", type="password")

        # Create Neo4j driver
        graph = Graph(uri, auth=(user, password))

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        driver = create_neo4j_driver(uri, user, password)

        if uploaded_file is not None:
            st.write("Filename:", uploaded_file.name)
        
            # Open the uploaded PDF file
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
            # Display the number of pages in the PDF
            st.write(f"Number of pages: {doc.page_count}")
        
            # Extract and display text from the first page
            page = doc[0]
            text = page.get_text("text")
            st.text_area("First Page Text", text)
        
            # Process text to extract nodes and relationships
            reports_df, regulators_df = process_pdf_text(text)
            
            # Save nodes and relationships to Neo4j
            save_nodes_and_relationships(graph, reports_df, regulators_df)
            st.success(f"PDF data imported into Neo4j: {uploaded_file.name}")

    elif choice == "Query OpenAI":
        st.header("Query OpenAI")
        context = st.text_area("Provide context for the question (optional):", "")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            print(data_from_db)
            data_from_db = []
            print(data_from_db)
            if question:
                driver = create_neo4j_driver(uri, user, password)
                if driver is None:
                    st.error("Failed to connect to Neo4j. Check your credentials and try again.")
                    return

                print(data_from_db)
                data_from_db = fetch_data_based_on_prompt(driver, question)
                print(f"main : {data_from_db}")
                formatted_data = ""
                if data_from_db and isinstance(data_from_db, list):
                    for record in data_from_db:
                        print(record)
                        formatted_data += str(record) + "\n"
                    print(f"Formatted Data: {formatted_data}")
                for record in data_from_db:
                    formatted_data += str(record) + "\n"
                print(formatted_data)
                if formatted_data:
                    answer = query_openai(formatted_data)
                    st.write(answer)

                else:
                    st.error("No data found in the database matching the query.")

                evaluation = evaluate_response(answer, context, question)
                print(evaluation)
                #st.write("Evaluation:", evaluation)
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()
