Prerequisites
Python 3.7 or higher: Make sure you have Python installed. You can download it from [Python.org](https://www.python.org/downloads/).
Git: Ensure Git is installed on your system. You can download it from [Git-SCM.com](https://git-scm.com/downloads).
Neo4j: Install Neo4j Desktop from [Neo4j Downloads](https://neo4j.com/docs/desktop-manual/current/installation/download-installation/).

Installation Steps
1. Clone the Repository
Open your terminal or command prompt and clone the repository from GitHub:

git clone https://github.com/SnehaBiswas/Capstone-Project.git
cd Capstone-Project


2. Set Up a Virtual Environment
Set up a virtual environment to manage dependencies:

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

If there are failures: [Solution](https://docs.google.com/document/d/1wTnGUctboAakESZ3heTvbFA7LeRTpe21Dc1otHFMi6w/edit) 

3. Install Dependencies
Install the required dependencies using pip:

pip install -r requirements.txt

4. Configure Neo4j
Ensure Neo4j is installed and running:

Open Neo4j Desktop.
Create a new project and database.
Start the database and note down the connection credentials (URI, username, password).

5. Configure the Streamlit App
Create a .env file in the project root directory with the following content, replacing with your Neo4j credentials:

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

6. Run the Streamlit App
Run the Streamlit app with the following command:

streamlit run app.py

7. Access the App
Open your web browser and go to http://localhost:8501 to access the Streamlit app.



Troubleshooting
Common Issues
Neo4j Connection Error:

Ensure Neo4j is running and the credentials in the .env file are correct.
Port Already in Use:

If port 8501 is already in use, specify a different port when running the app:

streamlit run app.py --server.port <new-port>
Dependencies Not Installing:

Ensure you are using a compatible Python version and the virtual environment is activated.
