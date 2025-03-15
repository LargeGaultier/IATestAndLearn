from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from transformers import pipeline

# Charger l'index FAISS
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
vector_index = FAISS.load_local("./data/faiss_index", HuggingFaceEmbeddings(model_name=embedding_model))

# Charger le modèle LLM local
generator_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# Pipeline de questions/réponses
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)

# Fonction de chat
def ask_question(question):
    result = qa_chain.run(question)
    return result

if __name__ == "__main__":
    while True:
        query = input("Pose ta question (ou tape 'exit' pour quitter) : ")
        if query.lower() == "exit":
            break
        print("Réponse :", ask_question(query))
