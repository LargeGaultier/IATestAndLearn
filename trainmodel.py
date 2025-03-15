# Exemple: création d’un index vectoriel FAISS avec LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 1. Charger et segmenter un PDF en chunks de texte
loader = PyPDFLoader("C:\Users\crabe\source\repos\SanjeevLearning\Data\COURTS ACT, Cap 168, (Act 41 of 1945).pdf")
pages = loader.load_and_split()  # découpage par page puis éventuellement par chunk

docs = [page.page_content for page in pages]  # textes à indexer
# 2. Créer les embeddings et l’index FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_index = FAISS.from_texts(docs, embedding_model)
vector_index.save_local("index_faiss")
print("Index vectoriel FAISS créé avec", len(docs), "chunks de texte.")
