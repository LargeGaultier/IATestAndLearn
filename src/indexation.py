from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

if __name__ == "__main__":
    text_path = "./data/extracted_text.txt"
    text = load_text(text_path)

    # Split en chunks
    chunks = split_text(text)

    # Créer l'index avec les embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = create_faiss_index(chunks, embedding_model)

    # Sauvegarde de l'index
    vector_index.save_local("./data/faiss_index")
    print("Index vectoriel FAISS créé et sauvegardé ✅")
