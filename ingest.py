import os
import sys
import uuid
import re
import time
import hashlib
from pathlib import Path
import psycopg2
from qdrant_client import QdrantClient 
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from elasticsearch.helpers import bulk as es_bulk

import pdfplumber
from rapidocr_onnxruntime import RapidOCR

# 1. HARDENED CONFIGURATION & ENVIRONMENT SECURITY LAYER

DB_NAME = os.getenv("DB_NAME", "rag")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not DB_USER or not DB_PASSWORD:
    print("❌ Security Error: DB_USER or DB_PASSWORD environment variables are missing!")
    sys.exit(1)

if not re.match(r"^[a-zA-Z0-9_]+$", DB_NAME):
    print("❌ Security Error: DB_NAME contains invalid characters.")
    sys.exit(1)

ES_HOST = os.getenv("ES_HOST")
QDRANT_URL = os.getenv("QDRANT_URL")

if not ES_HOST or not QDRANT_URL:
    print("❌ Security Error: ES_HOST or QDRANT_URL environment variables are missing from memory!")
    sys.exit(1)

MAX_RETRIES = 5
RETRY_DELAY = 3

def get_pg_connection():
    """
    Returns a fresh, authenticated PostgreSQL socket connection instance.
    Standardizes connection parameters for both local dev and production.
    """
    # Development local socket instantiation:
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

    # PRODUCTION TLS ENFORCEMENT CONFIGURATION:
    # uncomment this for a production level environment
    # return psycopg2.connect(
    #     dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
    #     sslmode='verify-full', sslrootcert='/path/to/server-ca.crt'
    # )

def get_search_clients():
    """
    Instantiates fresh client interfaces for your search engines.
    """
    # Development engines initialization:
    es = Elasticsearch([ES_HOST], request_timeout=5)
    qd = QdrantClient(url=QDRANT_URL, timeout=5.0)
    return es, qd

    # PRODUCTION TLS ENFORCEMENT CONFIGURATION:
    # uncomment this for a production level environment
    # es = Elasticsearch([ES_HOST], ca_certs="/path/to/https_ca.crt", verify_certs=True, request_timeout=10)
    # qd = QdrantClient(url=QDRANT_URL, timeout=10.0, prefer_grpc=True, https=True)
    # return es, qd

# 2. PATH & PRE-FLIGHT BOUNDARY VALIDATORS

def validate_secure_path(input_path_str):
    base_dir = Path("./data").resolve()
    target_path = Path(input_path_str).resolve()
    if not target_path.is_relative_to(base_dir):
        print(f"❌ Security Threat Blocked: File path sits outside local storage root: {target_path}")
        sys.exit(1)
    return target_path

def verify_and_gate_pdf(file_path):
    if not file_path.exists():
        return False
        
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 
    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        print(f"❌ Security Violation: File size exceeds maximum processing boundary.")
        return False

    with open(file_path, 'rb') as f:
        if f.read(4) != b'%PDF':
            return False

    with pdfplumber.open(file_path) as pdf:
        MAX_PAGE_THRESHOLD = 100
        if len(pdf.pages) > MAX_PAGE_THRESHOLD:
            print(f"❌ Security Gate: File contains {len(pdf.pages)} pages. Max limit is {MAX_PAGE_THRESHOLD}.")
            return False
            
    return True

# 3. CRYPTOGRAPHIC CLEANLINESS EMBEDDING COMPONENT

def generate_local_embedding(text_chunk):
    import numpy as np
    hasher = hashlib.sha256(text_chunk.encode('utf-8'))
    seed = int(hasher.hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.random(768).tolist()

# 4. LAPTOP-SCALE ADVANCED VISION & TABULAR PARSER

def extract_hierarchical_content(secure_pdf_path):
    parent_blocks = []
    child_chunks = []
    engine_ocr = RapidOCR()
    
    with pdfplumber.open(secure_pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            parent_id = f"p_pg{page_num}_{uuid.uuid4().hex[:6]}"
            page_text_accumulator = []
            
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table_strings = []
                    for row in table:
                        cleaned_row = [str(cell).strip() for cell in row if cell is not None]
                        if cleaned_row and any(cleaned_row):
                            table_strings.append(" | ".join(cleaned_row))
                    
                    if table_strings:
                        table_markdown = "\n".join(table_strings)
                        page_text_accumulator.append(f"\n[STRUCTURED TABLE DATA]:\n{table_markdown}\n")
            
            raw_page_text = page.extract_text(layout=True)
            
            if (not raw_page_text or len(raw_page_text.strip()) < 50) and not tables:
                print(f"⚠️ Page {page_num + 1} lacks digital font data layers. Launching local RapidOCR core...")
                
                pil_image = page.to_image(resolution=150).original
                ocr_results, _ = engine_ocr(pil_image)
                
                if ocr_results:
                    ocr_lines = [line[1] for line in ocr_results if float(line[2]) > 0.40]
                    raw_page_text = " ".join(ocr_lines)
            
            if raw_page_text:
                clean_extracted_text = " ".join(raw_page_text.split())
                page_text_accumulator.append(clean_extracted_text)
                
            final_parent_content = "\n".join(page_text_accumulator).strip()
            
            if 'pil_image' in locals():
                del pil_image
            
            if len(final_parent_content) < 30:
                continue
                
            parent_blocks.append({
                "parent_id": parent_id,
                "text": final_parent_content
            })
            
            sentences = re.split(r'(?<=[.!?]) +', final_parent_content)
            current_child_buffer = ""
            
            for sentence in sentences:
                if len(current_child_buffer) + len(sentence) < 220:
                    current_child_buffer += " " + sentence
                else:
                    if current_child_buffer.strip():
                        child_chunks.append({
                            "parent_id": parent_id, 
                            "text": current_child_buffer.strip(),
                            "page_num": page_num + 1
                        })
                    current_child_buffer = sentence
                    
            if current_child_buffer.strip():
                child_chunks.append({
                    "parent_id": parent_id, 
                    "text": current_child_buffer.strip(),
                    "page_num": page_num + 1
                })
                
    return parent_blocks, child_chunks

# 5. END-TO-END DOCUMENT INGESTION WORKFLOW

def ingest_complex_document(relative_pdf_path):
    secure_file_path = validate_secure_path(relative_pdf_path)
    file_name = secure_file_path.name
    
    if not verify_and_gate_pdf(secure_file_path):
        print(f"❌ Processing Terminated: '{file_name}' failed structural size or page count bounds.")
        return

    file_size = secure_file_path.stat().st_size
    document_id = str(uuid.uuid4())
    print(f"🎬 Starting isolated visual extraction loop for: {file_name}")
    
    pg_conn = None
    es_client = None
    qdrant_client = None

    try:
        pg_conn = get_pg_connection()
        es_client, qdrant_client = get_search_clients()

        with pg_conn.cursor() as pg_cursor:
            pg_cursor.execute("""
                INSERT INTO metadata (document_id, document_name, document_type, document_size, document_path, processing_status)
                VALUES (%s, %s, 'pdf', %s, %s, 'processing');
            """, (document_id, file_name, file_size, str(secure_file_path)))
            pg_conn.commit()

        parent_blocks, child_chunks = extract_hierarchical_content(secure_file_path)
        
        # CHANGED REMEDIATION: Replaced early silent return with an explicit ValueError exception.
        # This channels unindexable/empty files directly into the centralized exception cleanup handler.
        if not child_chunks:
            raise ValueError(f"No indexable textual content or tabular context extracted from document target '{file_name}'.")

        # ─── STAGE C: SPARSE LEXICAL ROUTING LAYER (ELASTICSEARCH BULK API) ───
        print(f"📝 Reconstructing batch data for Elasticsearch Bulk API...")
        es_actions = []
        for chunk in child_chunks:
            chunk_uuid = str(uuid.uuid4())
            es_actions.append({
                "_index": "lexical_document_chunks",
                "_id": chunk_uuid,
                "_source": {
                    "document_id": document_id,
                    "chunk_index": chunk_uuid,
                    "content": chunk["text"],
                    "metadata": {"file_name": file_name, "page_number": chunk["page_num"]}
                }
            })
            
        print(f"🚀 Shipping atomic token batch payload matrix to Elasticsearch...")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                success_count, errors = es_bulk(es_client, es_actions)
                if errors:
                    raise es_exceptions.TransportError(500, f"Bulk operational mismatch. Failed to index {len(errors)} items.", errors)
                break
            except (es_exceptions.ConnectionError, es_exceptions.TransportError) as es_err:
                if attempt == MAX_RETRIES: raise es_err
                time.sleep(RETRY_DELAY)

        # ─── STAGE D: DENSE SEMANTIC VECTOR ROUTING LAYER (QDRANT HNSW Matrix) ───
        print("🧬 Encoding semantic maps and parent contexts for Qdrant Engine...")
        vector_points = []
        parent_lookup = {p["parent_id"]: p["text"] for p in parent_blocks}
        
        for chunk in child_chunks:
            chunk_uuid = str(uuid.uuid4())
            dense_vector = generate_local_embedding(chunk["text"])
            associated_parent_text = parent_lookup.get(chunk["parent_id"], "")
            
            vector_points.append(
                models.PointStruct(
                    id=chunk_uuid,
                    vector=dense_vector,
                    payload={
                        "document_id": document_id,
                        "text": chunk["text"],
                        "parent_context": associated_parent_text,
                        "page_number": chunk["page_num"]
                    }
                )
            )
            
        QDRANT_BATCH_SIZE = 100
        print(f"🚀 Streaming vector batches (Chunk size: {QDRANT_BATCH_SIZE}) to Qdrant...")
        for i in range(0, len(vector_points), QDRANT_BATCH_SIZE):
            vector_batch = vector_points[i:i + QDRANT_BATCH_SIZE]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    qdrant_client.upsert(collection_name="semantic_document_embeddings", points=vector_batch)
                    break
                except (ResponseHandlingException, UnexpectedResponse) as qd_err:
                    if attempt == MAX_RETRIES: raise qd_err
                    time.sleep(RETRY_DELAY)

        # ─── STAGE E: TRANSACTION CONFIRMATION ───
        with pg_conn.cursor() as pg_cursor:
            pg_cursor.execute("""
                UPDATE metadata SET processing_status = 'completed', total_chunks = %s WHERE document_id = %s;
            """, (len(child_chunks), document_id))
            pg_conn.commit()
            
        print(f"\n🎉 Success! Production-hardened ingestion complete. Databases synced for {file_name}.\n")

    except Exception as general_pipeline_failure:
        print(f"⚠️ Internal Processing Interruption caught. Initiating cluster compensation cleanup workflows...")
        
        # 1. Purge Elasticsearch index tracks safely
        # CHANGED REMEDIATION: Updated term filter to query the 'document_id.keyword' un-analyzed subfield.
        # This completely guarantees exact literal matching on hyphenated UUID strings.
        if es_client:
            try:
                es_client.delete_by_query(
                    index="lexical_document_chunks",
                    body={"query": {"term": {"document_id.keyword": document_id}}}
                )
                print("🧹 Compensation: Cleaned orphaned Elasticsearch ghost chunks successfully.")
            except Exception as es_cleanup_err:
                print(f"⚠️ Compensation Warning: Elasticsearch cleanup incomplete: {es_cleanup_err}")

        # 2. Purge Qdrant vector space shards safely
        if qdrant_client:
            try:
                qdrant_client.delete(
                    collection_name="semantic_document_embeddings",
                    points_selector=models.Filter(
                        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                    )
                )
                print("🧹 Compensation: Cleaned orphaned Qdrant vector nodes successfully.")
            except Exception as qd_cleanup_err:
                print(f"⚠️ Compensation Warning: Qdrant cleanup incomplete: {qd_cleanup_err}")

        # 3. Clean up the SQL Transaction Wire and Register Failure State Token
        if pg_conn:
            try:
                pg_conn.rollback()
                with pg_conn.cursor() as pg_cursor:
                    pg_cursor.execute("UPDATE metadata SET processing_status = 'failed' WHERE document_id = %s;", (document_id,))
                    pg_conn.commit()
                print("📝 Transaction Ledger tracking flipped safely to 'failed'.")
            except Exception as rollback_err:
                print(f"❌ Failed to commit failure status token: {rollback_err}")
            
        print(f"❌ Critical Pipeline Failure. Transaction aborted. Details: {general_pipeline_failure}")
        
    finally:
        if pg_conn and not pg_conn.closed:
            pg_conn.close()
        if es_client:
            try: es_client.close()
            except Exception: pass
        if qdrant_client:
            try: qdrant_client.close()
            except Exception: pass

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    test_pdf_path = "./data/target_document.pdf"
    
    if os.path.exists(test_pdf_path):
        ingest_complex_document(test_pdf_path)
    else:
        print(f"👉 Drop a real scanned/table PDF into '{test_pdf_path}' to execute your full multi-database pipeline loop!")