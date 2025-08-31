"""
Script to populate the Neo4j database with patient data from a CSV
and clinical notes from markdown files, including LLM-based entity extraction
and linking, plus chunking for vector search.
"""
import os
import pandas as pd
import json
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
from tqdm import tqdm
from .connection import driver

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM for entity extraction
llm = VertexAI(model_name="gemini-2.5-flash", temperature=0)

def ingest_csv_data():
    """
    Loads patient data from the CSV file, creates Patient and Diagnosis nodes,
    and establishes relationships between them in Neo4j.
    """
    print("Ingesting data from patient_data.csv...")
    df = pd.read_csv('data/patient_data.csv')

    # Ensure all columns are treated as strings for consistent processing
    df = df.astype(str)
    patient_records = df.to_dict('records')

    ingest_query = """
    UNWIND $records AS record
    MERGE (p:Patient {patient_id: record.patient_id})
    ON CREATE SET
        p.age = toInteger(record.age),
        p.sex = record.sex,
        p.bmi = toFloat(record.bmi),
        p.smoker = record.smoker,
        p.medication_count = toInteger(record.medication_count),
        p.days_hospitalized = toInteger(record.days_hospitalized),
        p.readmitted = toInteger(record.readmitted),
        p.last_lab_glucose = toFloat(record.last_lab_glucose),
        p.exercise_frequency = record.exercise_frequency,
        p.diet_quality = record.diet_quality,
        p.income_bracket = record.income_bracket,
        p.education_level = record.education_level,
        p.urban = toInteger(record.urban),
        p.albumin_globulin_ratio = toFloat(record.albumin_globulin_ratio),
        p.chronic_obstructive_pulmonary_disease = record.chronic_obstructive_pulmonary_disease,
        p.alanine_aminotransferase = toFloat(record.alanine_aminotransferase)
    MERGE (d:Diagnosis {code: record.diagnosis_code})
    MERGE (p)-[:HAS_DIAGNOSIS]->(d)
    """

    with driver.session() as session:
        session.run(ingest_query, records=patient_records)

    print("CSV data ingestion complete.")

def extract_entities_from_document(doc):
    """
    Uses an LLM to extract structured entities from a clinical document.
    """
    extraction_prompt = f"""
    From the clinical document below, extract the following information:
    - patient_id: The patient's unique identifier.
    - patient_name: The full name of the patient.
    - patient_gender: The gender of the patient.
    - patient_age: The patient's age. Calculate it from the Date of Birth if possible, otherwise look for an explicit age.
    - symptoms: A list of key symptoms or complaints. Look in the "Chief Complaint" and "History of Present Illness" sections.
    - diagnoses: A list of JSON objects, where each object has a "code" and a "description" key. For example: [{{"code": "R05", "description": "Cough"}}].    - treatment_plan: A concise summary of the "Treatment Plan" section.
    - medications: A list of all prescribed medications (just the medication names).
    - clinician_name: The name of the primary clinician.
    - clinician_specialization: The specialization of the clinician.
    - temperature: The patient's temperature value as a float.
    - blood_pressure: The patient's blood pressure as a string (e.g., "128/82").
    - heart_rate: The patient's heart rate as an integer.
    - respiratory_rate: The patient's respiratory rate as an integer.
    - oxygen_saturation: The patient's oxygen saturation as a float.

    Return the information as a valid JSON object with these exact keys. If any information is not found, use null for that field.
    Do not include any other text or explanation.

    DOCUMENT:
    {doc.page_content}
    """

    try:
        response = llm.invoke(extraction_prompt)
        # The response is often wrapped in markdown, so we clean it
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error processing document {doc.metadata.get('source', 'Unknown')}: {e}")
        print(f"LLM Response was: {response}")
        return None

def ingest_markdown_data():
    """
    Loads markdown files, extracts entities using an LLM, links them to the graph,
    then splits the documents into chunks, generates embeddings, and stores
    them in a Neo4j vector index.
    """
    print("Ingesting data from markdown documents...")

    loader = DirectoryLoader(
        'documents_data/markdowns/',
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    documents = loader.load()

    print("Extracting and linking entities from documents...")

    with driver.session() as session:
        # Create a progress bar for the document processing loop
        # This tracks the most time-consuming part: LLM extraction + database operations
        for doc in tqdm(documents, desc="Processing documents", unit="doc"):
            entities = extract_entities_from_document(doc)
            if entities and entities.get('patient_id'):
                # Fixed query with proper subquery syntax
                link_query = """
                // Part 1: Upsert the Patient and set properties.
                MERGE (p:Patient {patient_id: $patient_id})
                ON CREATE SET
                    p.name = $patient_name,
                    p.sex = $patient_gender,
                    p.age = $patient_age
                // Use SET to update vitals. COALESCE prefers the new value if it's not null.
                SET
                    p.temperature = COALESCE($temperature, p.temperature),
                    p.blood_pressure = COALESCE($blood_pressure, p.blood_pressure),
                    p.heart_rate = COALESCE($heart_rate, p.heart_rate),
                    p.respiratory_rate = COALESCE($respiratory_rate, p.respiratory_rate),
                    p.oxygen_saturation = COALESCE($oxygen_saturation, p.oxygen_saturation)

                // Part 2: Upsert the Document, link it, and add the treatment plan to the relationship.
                MERGE (doc:Document {source: $source})
                MERGE (p)-[r:HAS_DOCUMENT]->(doc)
                SET r.treatment_plan = $treatment_plan

                // Part 3: Carry the patient forward and perform conditional linking using subqueries.
                WITH p

                // Conditionally link and enrich multiple Diagnoses
                CALL {
                    WITH p
                    // Unwind the list of diagnosis objects passed in the parameters
                    WITH p, $diagnoses AS diagnoses_list
                    WHERE diagnoses_list IS NOT NULL AND size(diagnoses_list) > 0
                    UNWIND diagnoses_list AS diagnosis
                    // Process each diagnosis object from the list
                    WITH p, diagnosis WHERE diagnosis.code IS NOT NULL AND diagnosis.code <> ""
                    MERGE (diag:Diagnosis {code: diagnosis.code})
                    // Set the description from the object
                    SET diag.description = diagnosis.description
                    MERGE (p)-[:HAS_DIAGNOSIS]->(diag)
                }

                // Conditionally link Clinician
                CALL {
                    WITH p
                    WITH p, $clinician_name AS clin_name, $clinician_specialization AS clin_spec
                    WHERE clin_name IS NOT NULL AND clin_name <> ""
                    MERGE (clin:Clinician {name: clin_name})
                    ON CREATE SET clin.specialization = clin_spec
                    MERGE (p)-[:TREATED_BY]->(clin)
                }

                // Conditionally link Medications
                CALL {
                    WITH p
                    WITH p, $medications AS med_list
                    WHERE med_list IS NOT NULL AND size(med_list) > 0
                    UNWIND med_list AS med_name
                    WITH p, med_name WHERE med_name IS NOT NULL AND med_name <> ""
                    MERGE (med:Medication {name: med_name})
                    MERGE (p)-[:PRESCRIBED]->(med)
                }

                // Conditionally link Symptoms
                CALL {
                    WITH p
                    WITH p, $symptoms AS symptom_list
                    WHERE symptom_list IS NOT NULL AND size(symptom_list) > 0
                    UNWIND symptom_list AS symptom_name
                    WITH p, symptom_name WHERE symptom_name IS NOT NULL AND symptom_name <> ""
                    MERGE (symp:Symptom {name: symptom_name})
                    MERGE (p)-[:HAS_SYMPTOM]->(symp)
                }
                """

                params = {
                    "source": doc.metadata.get('source'),
                    "patient_id": entities.get('patient_id'),
                    "patient_name": entities.get('patient_name'),
                    "patient_gender": entities.get('patient_gender'),
                    "patient_age": entities.get('patient_age'),
                    "symptoms": entities.get('symptoms', []),
                    "diagnoses": entities.get('diagnoses', []),
                    "treatment_plan": entities.get('treatment_plan'),
                    "clinician_name": entities.get('clinician_name'),
                    "clinician_specialization": entities.get('clinician_specialization'),
                    "medications": entities.get('medications', []),
                    "temperature": entities.get('temperature'),
                    "blood_pressure": entities.get('blood_pressure'),
                    "heart_rate": entities.get('heart_rate'),
                    "respiratory_rate": entities.get('respiratory_rate'),
                    "oxygen_saturation": entities.get('oxygen_saturation'),
                }

                session.run(link_query, params)

    print("Splitting documents and creating vector embeddings...")

    # Split documents into chunks for vector search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_for_vector = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

    # Create vector store in Neo4j
    Neo4jVector.from_documents(
        docs_for_vector,
        embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database="neo4j",
        index_name="documents",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
        create_id_index=True,
    )

    print("Markdown ingestion and relationship creation complete.")

if __name__ == "__main__":
    print("Clearing database...")

    with driver.session() as session:
        # Clear all existing data
        session.run("MATCH (n) DETACH DELETE n")

        # Drop existing vector index if it exists
        try:
            session.run("CALL db.index.vector.drop('documents')")
            print("Dropped existing vector index.")
        except Exception:
            print("No existing vector index to drop.")

    # Execute the data ingestion pipeline
    ingest_csv_data()
    ingest_markdown_data()

    print("Database population finished.")
    driver.close()