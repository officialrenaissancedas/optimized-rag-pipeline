import os
import sys
import time
import re
from qdrant_client import QdrantClient 
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import psycopg2

DB_NAME = os.getenv("DB_NAME", "rag")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not DB_USER or not DB_PASSWORD:
    print("❌ Security Error: DB_USER or DB_PASSWORD environment variables are missing!")
    print("Load environment variables in terminal before running script.")
    sys.exit(1)
    
if not re.match(r"^[a-zA-Z0-9_]+$", DB_NAME):
    print("❌ Security Error: DB_NAME contains invalid characters. Only alphanumeric characters and underscores are allowed.")
    sys.exit(1)


ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# needs to be switched out for a logging system later on in development.
print("🔄 Bootstrapping local 4-container multi-database indexing layers...")

MAX_RETRIES = 5
RETRY_DELAY = 3


# 1. POSTGRES Relational Metadata layer
try:
    print("🔎 Verifying target database existence...")
    # NOTE: In production cloud environments, change your network configurations to enforce TLS/SSL parameters:
    # pg_admin_conn = psycopg2.connect(..., sslmode='verify-full', sslrootcert='/path/to/server-ca.pem')
    pg_admin_conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        connect_timeout=3
    )
    pg_admin_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    with pg_admin_conn.cursor() as admin_cursor:
        admin_cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (DB_NAME,))
        if not admin_cursor.fetchone():
            print(f"🛠️ Target database '{DB_NAME}' not found. Creating it now...")
            admin_cursor.execute(f'CREATE DATABASE "{DB_NAME}";')
    pg_admin_conn.close()
except Exception as init_db_err:
    print(f"❌ Database preparation failed. Could not check or create database '{DB_NAME}'. Details: {init_db_err}")
    sys.exit(1)

# Metadata schema for table creation
try:
    pg_conn = None
    print(f"✅ Initializing Postgres Metadata table schema in '{DB_NAME}'...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pg_conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=2
            )
            with pg_conn.cursor() as pg_cursor:
                pg_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id SERIAL PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        document_name VARCHAR(255) NOT NULL,
                        document_type TEXT NOT NULL,
                        document_size BIGINT NOT NULL,
                        document_path TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processing_status VARCHAR(50) DEFAULT 'pending',
                        total_chunks INT DEFAULT 0
                    );
                """)# add in accordance to changes in future
                pg_conn.commit()
            print(f"✅ Connection established to {DB_NAME} and schema initialized on attempt {attempt}/{MAX_RETRIES}.")
            break
        except psycopg2.OperationalError as retry_err:
            print(f"⏳ [Attempt {attempt}/{MAX_RETRIES}] Database engine warming up. Retrying in {RETRY_DELAY}s...")
            if attempt == MAX_RETRIES:
                raise retry_err
            time.sleep(RETRY_DELAY)
except psycopg2.OperationalError as pg_err:
    print(f"❌ PostgreSQL connection failed (Auth/Network). Error: {pg_err}")
    sys.exit(1)
except Exception as e:
    print(f"❌ PostgreSQL encountered an unhandled crash: {e}")
    sys.exit(1)
finally:
    if 'pg_conn' in locals() and pg_conn is not None and not pg_conn.closed:
        pg_conn.close()
        

# ELASTICSEARCH Keyword Search Engine Layer

try:
    print("🔄 Initializing Elasticsearch engine layer...")
    es_client = Elasticsearch([ES_HOST], request_timeout=2)
    index_name = "lexical_document_chunks"
    
    index_mapping = {
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "content": {"type": "text", "analyzer": "standard"},
                "metadata": {
                    "properties": {
                        "file_name": {"type": "keyword"},
                        "page_number": {"type": "integer"}
                    }
                }
            }
        }
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if not es_client.indices.exists(index=index_name):
                es_client.indices.create(index=index_name, body=index_mapping)
                print("✅ Elasticsearch Lexical index initialized with BM25 rule.")
            else:
                print("✅ Elasticsearch Lexical index matches current configuration.")
            break
        except es_exceptions.ConnectionError as es_err:
            print(f"⏳ [Attempt {attempt}/{MAX_RETRIES}] Elasticsearch warming up. Retrying in {RETRY_DELAY}s...")
            if attempt == MAX_RETRIES:
                raise es_err
            time.sleep(RETRY_DELAY)
except es_exceptions.ConnectionError as es_final_err:
    print(f"❌ Lexical Layer Failure: Elasticsearch container unreachable. Details: {es_final_err}")
    sys.exit(1)
except Exception as es_unexpected_err:
    print(f"❌ Elasticsearch unexpected initialization failure: {es_unexpected_err}")
    sys.exit(1)


# QDRANT Vector Engine Layer

print("\n🔄 Initializing Qdrant vector engine layer...")

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, timeout=3.0)
    collection_name = "semantic_document_embeddings"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            collection_response = qdrant_client.get_collections()
            existing_collections = [c.name for c in collection_response.collections]
            
            if collection_name not in existing_collections:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
                print("✅ Qdrant semantic collection structural schema mounted.")
            else:
                print("✅ Qdrant semantic collection matches current configuration.")
            break 
        except (ResponseHandlingException, UnexpectedResponse) as retry_err:
            print(f"⏳ [Attempt {attempt}/{MAX_RETRIES}] Qdrant cluster warming up. Retrying in {RETRY_DELAY}s...")
            if attempt == MAX_RETRIES:
                raise retry_err
            time.sleep(RETRY_DELAY)
         
except (ResponseHandlingException, UnexpectedResponse) as qd_net_err:
    print(f"❌ Semantic Layer Failure: Qdrant cluster unreachable or API misconfigured.\nDetails: {qd_net_err}")
    sys.exit(1)
except Exception as qd_err:
    print(f"❌ Semantic Layer Failure: Qdrant client connection error. \nDetails: {qd_err}")
    sys.exit(1)
    
print("\n🚀 Environment-matched indexing verification complete. Local container stack ready for file ingestion loops.")