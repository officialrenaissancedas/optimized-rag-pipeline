import os
import sys 
import re  
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
import psycopg2

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# use venv\Scripts\activate in cmd prompt to activate virtual environment before running this script
# use .\venv\Scripts\Activate.ps1 in powershell to activate virtual environment before running this script
# use python test_connections.py to run this 
# use docker-compose up -d to start the containers before running this script
print("Infrastructure connection testing\n")

qdrant_url = os.getenv("QDRANT_URL")
es_host = os.getenv("ES_HOST")

if not qdrant_url or not es_host:
    print("❌ Security Error: QDRANT_URL or ES_HOST network environment variables are missing from memory!")
    sys.exit(1)

# 1. TESTING QDRANT (VECTOR DB)

try:
    qdrant_client = QdrantClient(url=qdrant_url)
    
    qdrant_client.get_collections()
    print("✅ Qdrant connection successful")

except (ResponseHandlingException, UnexpectedResponse) as e:
    print(f"❌ Qdrant connection failed (Network/API). Error: {e}")
except Exception as e:
    print(f"❌ Qdrant hit an unexpected framework crash. Error: {e}")

# 2. TESTING ELASTICSEARCH (KEYWORD SEARCH ENGINE)

try:
    # SYSTEM UPGRADE PROFILE: 
    # In a production environment, uncomment this line to pass CA file path:
    # es_client = Elasticsearch(hosts=es_host, ca_certs="/path/to/https_ca.crt")
    # In a development environment, uncomment this line to ignore SSL certificate verification: (Change .env cnfiguration for security changes)
    es_client = Elasticsearch(hosts=es_host)
    es_client.info()
    
    print("✅ Elasticsearch connection successful")
except ESConnectionError as e: 
    print(f"❌ Elasticsearch connection failed (Network/Timeout). Error: {e}")
except Exception as e:
    print(f"❌ Elasticsearch hit an unexpected error. Error: {e}")
    
# 3. TESTING POSTGRESQL (RELATIONAL METADATA DB)

try:
    db_name = os.getenv("DB_NAME", "rag")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    if not db_user or not db_password:
        print("❌ Security Error: DB_USER or DB_PASSWORD environment variables are missing from memory!")
        sys.exit(1)

    if not re.match(r"^[a-zA-Z0-9_]+$", db_name):
        print("❌ Security Error: DB_NAME contains invalid characters.")
        sys.exit(1)

    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password, 
        host=db_host,  
        port=db_port,
        connect_timeout=2 
    )
    print("✅ PostgreSQL connection successful")
    conn.close()
except psycopg2.OperationalError as e: 
    print(f"❌ PostgreSQL connection failed (Auth/Network). Error: {e}") 
except Exception as e:
    print(f"❌ PostgreSQL encountered an unhandled crash: {e}")