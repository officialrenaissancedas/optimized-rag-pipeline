import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
import psycopg2

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

# use venv\Scripts\activate in cmd prompt to activate virtual environment before running this script
# use .\venv\Scripts\Activate.ps1 in powershell to activate virtual environment before running this script
# use python test_connections.py to run this 
# use docker-compose up -d to start the containers before running this script
print("Infrastructure connection testing")

#testing qdrant (vector db)

try:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_client = QdrantClient(url=qdrant_url)
    qdrant_client.get_collections()
    print("Qdrant connection successful")
except Exception as e:
    print(f"Qdrant connection failed. Error: {e}")

#testing elasticsearch (keyword search engine)

try:
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    # SYSTEM UPGRADE PROFILE: 
    # In a production environment, uncomment this line to pass CA file path:
    # es_client = Elasticsearch(hosts=es_host, ca_certs="/path/to/https_ca.crt")
    # In a development environment, uncomment this line to ignore SSL certificate verification: (Change .env cnfiguration for security changes)
    es_client = Elasticsearch(hosts=es_host)
    info = es_client.info()
    print("Elasticsearch connection successful")
    print(f"Cluster Name: {info['cluster_name']}")
except ESConnectionError as e: 
    print(f"Elasticsearch connection failed (Network/Timeout). Error: {e}")
except Exception as e:
    print(f"Elasticsearch hit an unexpected error. Error: {e}")
    
#testing postgresql (relational metadata db)

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), 
        host=os.getenv("DB_HOST"),  
        port=os.getenv("DB_PORT")
    )
    print("PostgreSQL connection successful")
    conn.close()
except psycopg2.OperationalError as e: # FIX: Precise database engine exception check
    print(f"PostgreSQL connection failed (Auth/Network). Error: {e}") 
except Exception as e:
    print(f"PostgreSQL encountered an unhandled crash: {e}") 
        