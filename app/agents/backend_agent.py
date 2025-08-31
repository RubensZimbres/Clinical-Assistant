"""
Defines the backend agent's tools for querying the graph database,
running document retrieval, and making predictions.
"""
import os
import ast
from langchain_neo4j import GraphCypherQAChain
from langchain.chains import RetrievalQA  # Added missing import
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
# Import the prediction function from our new models module
from app.models.predict import predict_patient_outcomes

# Initialize the LLM
llm = VertexAI(model_name="gemini-2.5-flash", temperature=0)

# Initialize Neo4j graph connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Define the comprehensive clinical Cypher generation prompt
CLINICAL_CYPHER_PROMPT = PromptTemplate(
    template="""
Let's think step by step:

Step 1: Task:
Generate an effective and concise Cypher statement with less than 256 characters to query a clinical graph database.
Do not comment the code.

Step 2: Get to know the database schema: {schema}

Step 3: Instructions:
- In the cypher query, ONLY USE the provided relationship types and properties that appear in the schema AND in the user question.
- In the cypher query, do not use any other relationship types or properties in the user's question that are not contained in the provided schema.
- Regarding Age, NEVER work with exact age values. For example: instead of "24 years old", use intervals like "greater than 20 years old".
- USE ONLY ONE condition for Age, always use 'greater than' (>), never 'less than' or 'equal'.
- DO NOT USE property keys that are not in the database schema.
- Patient is the central node - most queries should start with or include Patient nodes.

Step 4: Examples:
Here are examples of generated Cypher statements for particular clinical questions:

4.1 Which patients have diabetes?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
WHERE d.code CONTAINS 'E11' OR d.description CONTAINS 'diabetes'
RETURN p.patient_id, p.name

4.2 What medications are prescribed for patients with high blood pressure?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
MATCH (p)-[:PRESCRIBED]->(m:Medication)
WHERE d.description CONTAINS 'hypertension' OR p.blood_pressure CONTAINS 'High'
RETURN DISTINCT m.name

4.3 Which patients over 50 have both diabetes and heart conditions?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d1:Diagnosis)
MATCH (p)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
WHERE p.age > 50 AND d1.description CONTAINS 'diabetes' AND d2.description CONTAINS 'heart'
RETURN p.patient_id, p.name, p.age

4.4 What symptoms do patients with appendicitis typically have?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE d.code = 'K35.80' OR d.description CONTAINS 'appendicitis'
RETURN s.name, COUNT(*) as frequency ORDER BY frequency DESC

4.5 Which clinicians treat the most patients with respiratory conditions?
MATCH (p:Patient)-[:TREATED_BY]->(c:Clinician)
MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
WHERE d.description CONTAINS 'respiratory' OR d.description CONTAINS 'lung'
RETURN c.name, c.specialization, COUNT(DISTINCT p) as patient_count ORDER BY patient_count DESC

4.6 What is the average BMI of patients with cardiovascular diagnoses?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
WHERE d.description CONTAINS 'cardiovascular' OR d.description CONTAINS 'heart'
RETURN AVG(p.bmi) as avg_bmi

4.7 What medications was the heart attack patient taking?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
WHERE toLower(d.description) CONTAINS 'heart attack'
   OR toLower(d.description) CONTAINS 'myocardial infarction'
   OR toLower(d.description) CONTAINS 'mi'
   OR d.code STARTS WITH 'I21'  // ICD-10 codes for acute MI
   OR d.code STARTS WITH 'I22'  // ICD-10 codes for subsequent MI
MATCH (p)-[:PRESCRIBED]->(m:Medication)
RETURN p.patient_id, p.name, d.description as diagnosis, m.name as medication
ORDER BY p.patient_id

4.8 What are the symptoms of seasonal allergies?
MATCH (p:Patient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
WHERE d.description =~ "(?i).*allerg.*"
MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.description as AllergyDiagnosis,
       collect(DISTINCT s.name) as Symptoms;

Step 5: Available Node Properties:
Patient properties: patient_id, name, age, sex, bmi, smoker, medication_count, days_hospitalized, readmitted, last_lab_glucose, exercise_frequency, diet_quality, income_bracket, education_level, urban, albumin_globulin_ratio, chronic_obstructive_pulmonary_disease, alanine_aminotransferase, temperature, blood_pressure, heart_rate, respiratory_rate, oxygen_saturation
Diagnosis properties: code, description
Medication properties: name
Symptom properties: name
Clinician properties: name, specialization
Document properties: source

Step 6: Available Relationship Types:
- Patient-[:HAS_DIAGNOSIS]->Diagnosis
- Patient-[:PRESCRIBED]->Medication
- Patient-[:HAS_SYMPTOM]->Symptom
- Patient-[:TREATED_BY]->Clinician
- Patient-[:HAS_DOCUMENT]->Document (with treatment_plan property)

Step 7: Query Guidelines:
- Use CONTAINS for text matching when searching descriptions or names
- Use exact matches (=) for codes and IDs
- For numerical comparisons, use >, <, >= operators
- When looking for patterns across multiple patients, use COUNT() and GROUP BY
- Consider using DISTINCT when aggregating to avoid duplicates
- For demographic queries, remember urban is encoded as 0/1 (rural/urban)

Step 8: Answer the question: {question}
Return only the Cypher query, with no additional text or explanation.
""",
    input_variables=["schema", "question"],
)

def setup_graph_qa_tool():
    """
    Sets up the tool for querying the structured graph data using our
    specialized clinical Cypher generation prompt.
    """
    graph_qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=CLINICAL_CYPHER_PROMPT,
        verbose=True,
        # Acknowledge the security risks of LLM-generated queries
        allow_dangerous_requests=True
    )

    return Tool(
        name="GraphQA",
        func=graph_qa_chain.invoke,
        description="""
        Use this tool for questions about patient statistics, counts, properties, and relationships.
        For example: 'How many smokers are there?', 'How many males older than 40 were readmitted?',
        'What medications were prescribed to patient P00001?', 'Which patients have diabetes?'.
        Provide the entire question as input without modification.
        """,
    )

def setup_document_rag_tool():
    """Sets up the tool for retrieving information from unstructured documents."""
    neo4j_vector = Neo4jVector(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        embedding=VertexAIEmbeddings(model_name="text-embedding-005"),
        index_name="documents",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    retriever = neo4j_vector.as_retriever()

    # Create the RAG chain for document retrieval
    rag_qa_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    return Tool(
        name="DocumentRAG",
        func=rag_qa_chain.invoke,
        description="""
        Use this tool for questions about medical procedures, symptoms, treatment plans, or other qualitative information
        likely found in clinical discharge summaries.
        For example: 'What are the symptoms of seasonal allergies?', 'Summarize the treatment plan for diabetic patients'.
        Provide the entire question as input without modification.
        """,
    )

def setup_prediction_tool():
    """Sets up the tool for making clinical predictions."""
    def prediction_tool_wrapper(query: str):
        try:
            # The LLM will pass the dictionary as a string, so we safely parse it
            input_dict = ast.literal_eval(query)
            if not isinstance(input_dict, dict):
                return "Error: Input must be a dictionary."
            return predict_patient_outcomes(input_dict)
        except (ValueError, SyntaxError) as e:
            return f"Error: Invalid input format. Please provide a valid Python dictionary string. Details: {e}"

    return Tool(
        name="PatientOutcomePredictor",
        func=prediction_tool_wrapper,
        description="""
        Use this tool ONLY to predict clinical outcomes like 'chronic_obstructive_pulmonary_disease' (COPD) or
        'alanine_aminotransferase' (ALT) for a patient with specific features. The input to this tool MUST be a
        Python dictionary string with patient data. For example:
        "{'age': 55, 'sex': 'Male', 'bmi': 27.5, 'smoker': 'No', 'medication_count': 3, 'exercise_frequency': 'None', 'diet_quality': 'Poor'}"
        Do not use this for any other type of question.
        """
    )

# A list of all tools the agent can use
agent_tools = [
    setup_graph_qa_tool(),
    setup_document_rag_tool(),
    setup_prediction_tool()
]