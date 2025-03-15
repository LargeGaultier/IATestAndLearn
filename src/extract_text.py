import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    pdf_path = "data/OSHI v Alteo Agri Ltd.pdf"
    output_path = "./data/extracted_text.txt"

    extracted_text = extract_text_from_pdf(pdf_path)
    save_text_to_file(extracted_text, output_path)

    print(f"Texte extrait et sauvegardé dans : {output_path}")
