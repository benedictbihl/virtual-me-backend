import os
from dotenv import load_dotenv
import openai

from supabase.client import Client, create_client

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import SupabaseVectorStore


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

supabase_url = os.environ.get("SUPABASE_URL") or ""
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or ""
supabase: Client = create_client(supabase_url, supabase_key)

# Define the folder where the wav files are located
root_folder = os.getenv("ROOT_DIR") or ""
print("Root folder: ", root_folder)

file_path = root_folder + "/transformed_documents"

# Get the number of json files in the root folder and its sub-folders
print("Getting number of files to embed...")
num_files = sum(
    1
    for dirpath, dirnames, filenames in os.walk(file_path)
    for filename in filenames
    if filename.endswith(".json")
)
print("Number of files: ", num_files)

docs = []

for dirpath, dirnames, filenames in os.walk(file_path):
    for filename in filenames:
        if filename.endswith(".json"):
            loader = JSONLoader(
                file_path=file_path + "/" + filename,
                jq_schema='.questions_and_answers[] | tostring | gsub("{"; "") | gsub("}"; "") | gsub("\\""; "")',
            )
            docs += loader.load()


print("Creating embeddings...")
embeddings = OpenAIEmbeddings(client=openai)
vector_store = SupabaseVectorStore.from_documents(docs, embeddings, client=supabase)
print("Embeddings created!")
