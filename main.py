import os
import json
import click
import time
import signal

try:
    import fitz  # PyMuPDF
    PDF_LIB = 'pymupdf'
except ImportError:
    fitz = None
    PDF_LIB = None
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        PDF_LIB = 'pdfminer'
    except ImportError:
        extract_pages = None
        LTTextContainer = None

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'app', 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'app', 'output')

def extract_text_by_page(pdf_path):
    """
    Extract plain text from each page of a PDF.
    Returns: list of dicts: [{'page': 1, 'text': '...'}, ...]
    """
    results = []
    if PDF_LIB == 'pymupdf' and fitz is not None:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            results.append({'page': page_num + 1, 'text': text})
        doc.close()
    elif PDF_LIB == 'pdfminer' and extract_pages is not None:
        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            texts = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    texts.append(element.get_text())
            results.append({'page': page_num + 1, 'text': ''.join(texts)})
    else:
        raise ImportError("No supported PDF library found. Please install PyMuPDF or pdfminer.six.")
    return results

def extract_headings(pages, pdf_path=None):
    """
    Extract headings using heuristics: font size, weight, position, line length, repetition.
    Returns: dict with title and outline.
    Only implemented for PyMuPDF.
    """
    if PDF_LIB != 'pymupdf' or fitz is None or pdf_path is None:
        # Fallback: Use text heuristics only (very basic)
        outline = []
        for page in pages:
            lines = page['text'].splitlines()
            for line in lines:
                if 0 < len(line.strip()) < 60 and line.strip().istitle():
                    outline.append({
                        'level': 'H2',
                        'text': line.strip(),
                        'page': page['page']
                    })
        return {'title': outline[0]['text'] if outline else '', 'outline': outline}
    # PyMuPDF: Use font/position heuristics
    doc = fitz.open(pdf_path)
    font_stats = {}
    headings = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")['blocks']
        for b in blocks:
            if b['type'] != 0:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    if not text or len(text) > 120:
                        continue
                    size = span['size']
                    font = span['font']
                    flags = span['flags']
                    bbox = span['bbox']
                    is_bold = 'Bold' in font or (flags & 2)
                    is_italic = 'Italic' in font or (flags & 1)
                    center_x = (bbox[0] + bbox[2]) / 2
                    page_width = page.rect.width
                    is_centered = abs(center_x - page_width/2) < page_width * 0.15
                    # Collect font size stats
                    font_stats.setdefault(size, 0)
                    font_stats[size] += 1
                    # Heuristic: Large, bold, centered, short lines = likely heading
                    if size >= 14 and is_centered and len(text) < 60:
                        level = 'H1' if size >= 18 else 'H2'
                        headings.append({'level': level, 'text': text, 'page': page_num})
                    elif is_bold and len(text) < 60:
                        headings.append({'level': 'H3', 'text': text, 'page': page_num})
    # Title: largest, most frequent centered text on first page
    title = ''
    if headings:
        h1s = [h for h in headings if h['level'] == 'H1']
        if h1s:
            title = h1s[0]['text']
        else:
            title = headings[0]['text']
    doc.close()
    return {'title': title, 'outline': headings}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@click.command()
def process_pdfs():
    """Process PDF files from /app/input and save outline JSON to /app/output."""
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} does not exist.")
        return
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(INPUT_DIR, f))]
    if not files:
        print("No PDF files found in input directory.")
        return
    for filename in files:
        print(f"Processing: {filename}")
        pdf_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, base_name + '.json')
        try:
            # Set up 10-second timeout (Unix only)
            if hasattr(signal, 'signal'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            page_texts = extract_text_by_page(pdf_path)
            headings = extract_headings(page_texts, pdf_path=pdf_path)
            if hasattr(signal, 'alarm'):
                signal.alarm(0)
        except TimeoutException:
            print(f"Processing {filename} timed out.")
            headings = {'title': '', 'outline': []}
        except Exception as e:
            print(f"Failed to extract from {filename}: {e}")
            headings = {'title': '', 'outline': []}
        output_data = {
            "title": headings['title'],
            "outline": headings['outline']
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved outline JSON to {output_path}")

if __name__ == '__main__':
    process_pdfs() 