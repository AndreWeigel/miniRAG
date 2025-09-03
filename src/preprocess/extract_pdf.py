import fitz  # PyMuPDF
from src.preprocess.cleaning import clean_page

def extract_pages(pdf_path: str, header_regex=None, footer_regex=None):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text("text")
        txt = clean_page(txt, header_regex=header_regex, footer_regex=footer_regex)
        pages.append({"page": i+1, "text": txt})
    return pages
