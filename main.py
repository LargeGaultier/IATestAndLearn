import gradio as gr
from src.chatbot import ask_question

def chat_interface(question):
    return ask_question(question)

interface = gr.Interface(
    fn=chat_interface,
    inputs="text",
    outputs="text",
    title="🧑‍⚖️ Chatbot - Décisions de justice"
)

if __name__ == "__main__":
    interface.launch()
