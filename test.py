from neo4j import GraphDatabase
import app 

# Initialize the Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Define your prompt
user_prompt = "Find all reports related to financial stability"

# Fetch data based on the prompt
results = app.fetch_data_based_on_prompt(driver, user_prompt)
print(results)


