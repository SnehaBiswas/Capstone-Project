import streamlit as st
import fitz  # PyMuPDF
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship
import re
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai.error import OpenAIError
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to create a Neo4j driver
def create_neo4j_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

# Function to clear the database
def clear_database(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

# Function to save nodes and relationships to Neo4j
def save_nodes_and_relationships(df, regulators_df, uri, user, password):
    graph = Graph(uri, auth=(user, password))
    
    for _, row in df.iterrows():
        source_node = Node("Node", name=row['Source_Node'])
        target_node = Node("Node", name=row['Target_Node'])
        relationship_type = row['Relationship']
        
        graph.merge(source_node, "Node", "name")
        graph.merge(target_node, "Node", "name")
        
        relationship = Relationship(source_node, relationship_type, target_node)
        graph.merge(relationship)

    for _, row in regulators_df.iterrows():
        source_node = Node("Node", name=row['Source_Node'])
        target_node = Node("Node", name=row['Target_Node'])
        relationship_type = row['Relationship']
        
        graph.merge(source_node, "Node", "name")
        graph.merge(target_node, "Node", "name")
        
        relationship = Relationship(source_node, relationship_type, target_node)
        graph.merge(relationship)

def extract_entities_and_relationships(text):
    # Example implementation using a simple regex-based approach
    entities = []
    relationships = []

    headers = re.findall(r'\n([A-Z].*?:)\n', text)  # Find headers (assumes headers end with a colon)
    sections = re.split(r'\n[A-Z].*?:\n', text)[1:]  # Split text into sections

    for i, section in enumerate(sections):
        header = headers[i]
        lines = section.split('\n')
        for line in lines:
            if ':' in line:
                entity, relationship = line.split(':', 1)
                entities.append((header.strip(), entity.strip()))
                relationships.append((header.strip(), relationship.strip()))

    df = pd.DataFrame(relationships, columns=['Source_Node', 'Target_Node'])
    df['Relationship'] = 'describes'  # Example relationship type

    regulators_df = pd.DataFrame(entities, columns=['Source_Node', 'Target_Node'])
    regulators_df['Relationship'] = 'has_entity'  # Example relationship type

    return df, regulators_df

# Main function for the Streamlit app
def main():
    st.title("Streamlit App with Neo4j and OpenAI Integration")

    st.sidebar.title("Navigation")
    options = ["Home", "PDF Upload with Neo4j Integration", "Query OpenAI", "View Knowledge Graph", "Clear Database"]
    choice = st.sidebar.selectbox("Select a page", options)

    if choice == "Home":
        st.header("Welcome to the Streamlit App")
        st.write("This app integrates Neo4j and OpenAI to handle queries and provide insights.")

    elif choice == "PDF Upload with Neo4j Integration":
        st.header("PDF Upload with Neo4j Integration")

        st.sidebar.header("Neo4j Connection")
        uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
        user = st.sidebar.text_input("User", "neo4j")
        password = st.sidebar.text_input("Password", type="password")

        driver = create_neo4j_driver(uri, user, password)

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            st.write("Filename:", uploaded_file.name)
            
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text("text")
            
            st.text_area("Extracted Text", text[:1000])  # Display the first 1000 characters
            
            df, regulators_df = extract_entities_and_relationships(text)
            save_nodes_and_relationships(df, regulators_df, uri, user, password)
            st.success("PDF info saved to Neo4j as a Knowledge Graph.")

    elif choice == "Query OpenAI":
        st.header("Query OpenAI")
        user_input = st.text_area("Enter your question:")
        if st.button("Get Answer"):
            if user_input:
                answer = query_openai(user_input)
                st.write(answer)
            else:
                st.error("Please enter a question.")

    elif choice == "View Knowledge Graph":
        st.header("Knowledge Graph Visualization")

        st.sidebar.header("Neo4j Connection")
        uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
        user = st.sidebar.text_input("User", "neo4j")
        password = st.sidebar.text_input("Password", type="password")

        if st.button("Load Graph"):
            if uri and user and password:
                data = fetch_graph_data(uri, user, password)
                html_path = visualize_graph(data)

                if os.path.exists(html_path):
                    with open(html_path, "r") as file:
                        html_content = file.read()
                    components.html(html_content, height=600, scrolling=True)
                else:
                    st.error(f"Failed to load graph. File not found: {html_path}")
            else:
                st.error("Please provide Neo4j connection details.")

    elif choice == "Clear Database":
        st.header("Clear Neo4j Database")

        st.sidebar.header("Neo4j Connection")
        uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
        user = st.sidebar.text_input("User", "neo4j")
        password = st.sidebar.text_input("Password", type="password")

        if st.button("Clear Data"):
            if uri and user and password:
                clear_database(uri, user, password)
                st.success("Neo4j database cleared successfully.")
            else:
                st.error("Please provide Neo4j connection details.")

if __name__ == "__main__":
    main()
