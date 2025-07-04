import PyPDF2
import io

def parse_document(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        return extract_pdf_text(uploaded_file)
    elif file_type == "txt":
        return extract_txt_text(uploaded_file)
    else:
        return "Unsupported file format."

def extract_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_txt_text(uploaded_file):
    return uploaded_file.read().decode("utf-8").strip()