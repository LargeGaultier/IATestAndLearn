from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Charger l'index FAISS
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
vector_index = FAISS.load_local(
    "./data/faiss_index",
    HuggingFaceEmbeddings(model_name=embedding_model),
    allow_dangerous_deserialization=True
)

# Charger le modèle LLM local
generator_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=300)
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
