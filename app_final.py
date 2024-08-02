import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship
import openai
import os
from dotenv import load_dotenv
import re
import spacy
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
#from langchain.neo4j import Neo4jConnector
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


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


def create_neo4j_driver(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver
    except Exception as e:
        print(f"Failed to create Neo4j driver: {e}")

def save_pdf_info(driver, filename):
    with driver.session() as session:
        session.write_transaction(create_pdf_node, filename)

def create_pdf_node(tx, filename):
    tx.run("CREATE (p:PDF {name: $filename})", filename=filename)

def process_pdf_text(text):
    regulator_pattern = re.compile(r'Board of Governors of Federal Reserve System|Office of Comptroller of the Currency|Federal Deposit Insurance Corporation')
    regulators = regulator_pattern.findall(text)

    # Extract report details including purpose and frequency
    # report_pattern = re.compile(r'(\d+)\.\s*([\w\s\-]+):\s*([\w\s\-\'\&\,\.]+)\nPurpose-\s*(.+)\nFrequency-\s*(.+)')
    # reports = report_pattern.findall(text)



    report_pattern = re.compile(r'(\d+)\.\s*([\w\s\-]+):\s*([\w\s\-\'\&\,\.]+)')
    regulator_pattern = re.compile(r'(Board of Governors of Federal Reserve System|Office of Comptroller of the Currency|Federal Deposit Insurance Corporation)')
    
    reports = []
    regulators = []
    
    lines = text.split('\n')
    
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
        
        graph.merge(id_node, "ReportID", "name")
        graph.merge(code_node, "ReportCode", "name")
        graph.merge(name_node, "ReportName", "name")
        
        id_to_code_relationship = Relationship(id_node, "HAS_CODE", code_node)
        code_to_name_relationship = Relationship(code_node, "HAS_NAME", name_node)
        
        graph.merge(id_to_code_relationship)
        graph.merge(code_to_name_relationship)
    
    for index, row in regulators_df.iterrows():
        regulator_node = Node("Regulator", name=row['Regulator'])
        graph.merge(regulator_node, "Regulator", "name")
    
    print("Data imported into Neo4j graph database successfully.")

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

# def generate_cypher_query(prompt):
#     doc = nlp(prompt)
#     entities = [ent.text for ent in doc.ents]
#     if "explain" in prompt.lower() or "what is" in prompt.lower():
#         if entities:
#             entity_name = entities[0]
#             return f"MATCH (n:ReportCode {{name: '{entity_name}'}}) RETURN n"
#     if "list" in prompt.lower() or "show" in prompt.lower():
#         if "reports" in prompt.lower():
#             return "MATCH (r:ReportName) RETURN r.name AS ReportName"
#         elif "regulators" in prompt.lower():
#             return "MATCH (r:Regulator) RETURN r.name AS Regulator"
#     if "relationship between" in prompt.lower():
#         if len(entities) >= 2:
#             entity1, entity2 = entities[:2]
#             return f"""
#             MATCH (a {{name: '{entity1}'}})-[r]->(b {{name: '{entity2}'}})
#             RETURN a, r, b
#             """
#     if "find" in prompt.lower() or "details of" in prompt.lower():
#         if entities:
#             entity_name = entities[0]
#             return f"""
#             MATCH (r:ReportCode {{name: '{entity_name}'}})
#             OPTIONAL MATCH (r)-[:HAS_CODE]->(c:ReportCode)
#             OPTIONAL MATCH (c)-[:HAS_NAME]->(n:ReportName)
#             RETURN r, c, n
#             """
#     return None

def fetch_data_based_on_prompt(prompt):
    langchain_llm = OpenAI(api_key=openai.api_key)
    text_splitter = CharacterTextSplitter()
    neo4j_connector = Neo4jGraph(url=uri, username=user, password=password)
    
    prompt_template = PromptTemplate(input_variables=["query"], template="Translate the following English prompt into a Cypher query: {query}")
    chain = LLMChain(llm=langchain_llm, prompt=prompt_template)

    def query_neo4j(cypher_query):
        try:
            result = neo4j_connector.query(cypher_query)
            print(result)
            return result
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return []
    
    cypher_query = chain.run(query=prompt)
    print(cypher_query)
    result = query_neo4j(cypher_query)
    print(result)
    return result

def main():
    global uri, user, password, data_from_db
    st.title("Streamlit App with Neo4j and OpenAI Integration")

    st.sidebar.title("Navigation")
    options = ["Home", "PDF Upload with Neo4j Integration", "Query OpenAI"]
    choice = st.sidebar.selectbox("Select a page", options)
    context=""
    if choice == "Home":
        st.header("Welcome to the Streamlit App")
        st.write("This app integrates Neo4j and OpenAI to handle queries and provide insights.")

    elif choice == "PDF Upload with Neo4j Integration":
        st.header("PDF Upload with Neo4j Integration")

        st.sidebar.header("Neo4j Connection")
        uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
        user = st.sidebar.text_input("User", "neo4j")
        password = st.sidebar.text_input("Password", type="password")

        graph = Graph(uri, auth=(user, password))

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        driver = create_neo4j_driver(uri, user, password)

        if uploaded_file is not None:
            st.write("Filename:", uploaded_file.name)
        
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
            st.write(f"Number of pages: {doc.page_count}")
        
            page = doc[0]
            text = page.get_text("text")
            st.text_area("First Page Text", text)
        
            reports_df, regulators_df = process_pdf_text(text)
            
            save_nodes_and_relationships(graph, reports_df, regulators_df)
            st.success(f"PDF data imported into Neo4j: {uploaded_file.name}")

    elif choice == "Query OpenAI":
        st.header("Query OpenAI")
        #context = st.text_area("Provide context for the question (optional):", "")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                data_from_db = fetch_data_based_on_prompt(question)
                formatted_data = ""
                if data_from_db and isinstance(data_from_db, list):
                    for record in data_from_db:
                        formatted_data += str(record) + "\n"
                if formatted_data:
                    answer = query_openai(formatted_data)
                    st.write(answer)
                    evaluation = evaluate_response(answer, context, question)
                    print("Evaluation:", evaluation)
                else:
                    st.error("No data found in the database matching the query.")
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()
