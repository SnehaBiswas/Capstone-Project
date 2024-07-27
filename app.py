import streamlit as st
import fitz  # PyMuPDF
from neo4j import GraphDatabase

def main():
    st.title("PDF Upload and Display")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Open the uploaded PDF file
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # Display the number of pages in the PDF
        st.write(f"Number of pages: {doc.page_count}")
        
        # Extract and display text from the first page
        page = doc[0]
        text = page.get_text("text")
        st.text_area("First Page Text", text)


def create_neo4j_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

def save_pdf_info(driver, filename):
    with driver.session() as session:
        session.write_transaction(create_pdf_node, filename)

# Function to create a PDF node in Neo4j
def create_pdf_node(tx, filename):
    tx.run("CREATE (p:PDF {name: $filename})", filename=filename)

def main():
    st.title("PDF Upload with Neo4j Integration")

    # Input fields for Neo4j connection
    st.sidebar.header("Neo4j Connection")
    uri = st.sidebar.text_input("URI", "bolt://localhost:7687")
    user = st.sidebar.text_input("User", "neo4j")
    password = st.sidebar.text_input("Password", type="password")

    # Create Neo4j driver
    driver = create_neo4j_driver(uri, user, password)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write("Filename:", uploaded_file.name)
        
        # Save PDF info to Neo4j
        save_pdf_info(driver, uploaded_file.name)
        st.success(f"PDF info saved to Neo4j: {uploaded_file.name}")

if __name__ == "__main__":
    main()
