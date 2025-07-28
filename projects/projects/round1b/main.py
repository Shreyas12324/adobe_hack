import os
import json
import click
import time
import signal
from datetime import datetime
import re
from collections import defaultdict

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

# NLP imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

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
                    font_stats.setdefault(size, 0)
                    font_stats[size] += 1
                    if size >= 14 and is_centered and len(text) < 60:
                        level = 'H1' if size >= 18 else 'H2'
                        headings.append({'level': level, 'text': text, 'page': page_num})
                    elif is_bold and len(text) < 60:
                        headings.append({'level': 'H3', 'text': text, 'page': page_num})
    
    title = ''
    if headings:
        h1s = [h for h in headings if h['level'] == 'H1']
        if h1s:
            title = h1s[0]['text']
        else:
            title = headings[0]['text']
    doc.close()
    return {'title': title, 'outline': headings}

def preprocess_text(text):
    """Clean and normalize text for NLP processing."""
    text = re.sub(r'\s+', ' ', text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()

def extract_keywords(text, top_n=20):
    """Extract top keywords from text using simple frequency analysis."""
    words = preprocess_text(text).split()
    word_freq = defaultdict(int)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_freq[word] += 1
    
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

def calculate_relevance_score(section_text, query_text):
    """Calculate relevance score using keyword overlap and TF-IDF if available."""
    if not NLP_AVAILABLE:
        # Fallback: simple keyword matching
        section_keywords = set([word for word, _ in extract_keywords(section_text, 50)])
        query_keywords = set([word for word, _ in extract_keywords(query_text, 50)])
        overlap = len(section_keywords.intersection(query_keywords))
        return overlap / max(len(section_keywords), 1)
    
    # Use TF-IDF and cosine similarity
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        texts = [preprocess_text(section_text), preprocess_text(query_text)]
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        # Fallback to keyword matching
        section_keywords = set([word for word, _ in extract_keywords(section_text, 50)])
        query_keywords = set([word for word, _ in extract_keywords(query_text, 50)])
        overlap = len(section_keywords.intersection(query_keywords))
        return overlap / max(len(section_keywords), 1)

def find_relevant_sections(documents_data, persona, job_description):
    """Find and rank relevant sections across all documents."""
    query = f"{persona} {job_description}"
    all_sections = []
    
    for doc_name, doc_data in documents_data.items():
        pages = doc_data['pages']
        headings = doc_data['headings']
        
        # Group text by sections based on headings
        current_section = None
        current_text = ""
        
        for page in pages:
            page_num = page['page']
            page_text = page['text']
            
            # Find headings on this page
            page_headings = [h for h in headings['outline'] if h['page'] == page_num]
            
            if page_headings:
                # Save previous section if exists
                if current_section and current_text.strip():
                    relevance_score = calculate_relevance_score(current_text, query)
                    all_sections.append({
                        'document': doc_name,
                        'page': current_section['page'],
                        'section_title': current_section['text'],
                        'importance_rank': 0,  # Will be set later
                        'relevance_score': relevance_score,
                        'text': current_text.strip()
                    })
                
                # Start new section
                current_section = page_headings[0]  # Use first heading on page
                current_text = page_text
            else:
                # Continue current section
                current_text += " " + page_text
        
        # Save last section
        if current_section and current_text.strip():
            relevance_score = calculate_relevance_score(current_text, query)
            all_sections.append({
                'document': doc_name,
                'page': current_section['page'],
                'section_title': current_section['text'],
                'importance_rank': 0,
                'relevance_score': relevance_score,
                'text': current_text.strip()
            })
    
    # Rank sections by relevance score
    all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    for i, section in enumerate(all_sections):
        section['importance_rank'] = i + 1
    
    return all_sections

def create_subsections(sections, max_length=500):
    """Create refined subsections from relevant sections."""
    subsections = []
    
    for section in sections[:10]:  # Top 10 sections
        text = section['text']
        if len(text) > max_length:
            # Split into paragraphs and take most relevant ones
            paragraphs = text.split('\n\n')
            relevant_paragraphs = []
            
            for para in paragraphs:
                if len(para.strip()) > 50:  # Minimum paragraph length
                    relevance = calculate_relevance_score(para, f"{section['section_title']}")
                    relevant_paragraphs.append((para, relevance))
            
            # Take top paragraphs
            relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
            refined_text = " ".join([para for para, _ in relevant_paragraphs[:3]])
        else:
            refined_text = text
        
        subsections.append({
            'document': section['document'],
            'page': section['page'],
            'refined_text': refined_text[:1000]  # Limit length
        })
    
    return subsections

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@click.command()
@click.option('--persona', default='PhD Researcher in Computational Biology', 
              help='Persona description')
@click.option('--job', default='Prepare literature review on GNNs for Drug Discovery', 
              help='Job-to-be-done description')
def process_pdfs_for_persona(persona, job):
    """Process multiple PDFs and find relevant sections based on persona and job description."""
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} does not exist.")
        return
    
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(INPUT_DIR, f))]
    if not files:
        print("No PDF files found in input directory.")
        return
    
    if len(files) < 3:
        print(f"Found only {len(files)} PDFs. Need at least 3 for meaningful analysis.")
        return
    
    print(f"Processing {len(files)} PDFs for persona: {persona}")
    print(f"Job: {job}")
    
    documents_data = {}
    start_time = time.time()
    
    # Process all PDFs
    for filename in files:
        print(f"Processing: {filename}")
        pdf_path = os.path.join(INPUT_DIR, filename)
        
        try:
            # Set up 10-second timeout per PDF
            if hasattr(signal, 'signal'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            
            page_texts = extract_text_by_page(pdf_path)
            headings = extract_headings(page_texts, pdf_path=pdf_path)
            
            if hasattr(signal, 'alarm'):
                signal.alarm(0)
            
            documents_data[filename] = {
                'pages': page_texts,
                'headings': headings
            }
            
        except TimeoutException:
            print(f"Processing {filename} timed out.")
        except Exception as e:
            print(f"Failed to extract from {filename}: {e}")
    
    # Find relevant sections
    print("Finding relevant sections...")
    sections = find_relevant_sections(documents_data, persona, job)
    
    # Create subsections
    print("Creating subsections...")
    subsections = create_subsections(sections)
    
    # Prepare output
    output_data = {
        "metadata": {
            "persona": persona,
            "job": job,
            "documents": list(documents_data.keys()),
            "timestamp": datetime.now().isoformat(),
            "processing_time": time.time() - start_time
        },
        "sections": [
            {
                "document": s['document'],
                "page": s['page'],
                "section_title": s['section_title'],
                "importance_rank": s['importance_rank']
            }
            for s in sections[:20]  # Top 20 sections
        ],
        "subsections": subsections
    }
    
    # Save output
    output_path = os.path.join(OUTPUT_DIR, 'persona_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete! Found {len(sections)} relevant sections.")
    print(f"Results saved to: {output_path}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    process_pdfs_for_persona() 