# app/graph_db/connection.py
"""
Manages the connection to the Neo4j database.
"""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def get_neo4j_driver():
    """
    Establishes and returns a connection driver to the Neo4j database.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))

driver = get_neo4j_driver()