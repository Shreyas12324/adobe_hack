# PDF Outline Extractor - Connecting the Dots Challenge (Round 1A)

## Overview
This solution extracts a structured outline from a PDF, identifying the title and hierarchical headings (H1, H2, H3) along with their page numbers. It runs entirely offline, within strict time, memory, and model-size constraints.

## Features
- Extracts:  
  - Title  
  - Headings (H1, H2, H3)  
  - Page numbers  
- Offline and Dockerized
- Fast: ≤ 10s for 50-page PDFs
- Compatible with `linux/amd64` architecture
- No internet or GPU required

## Approach
We use `PyMuPDF` to extract both text and layout metadata like:
- Font size
- Font weight
- Position on page

Heuristics are applied to classify headings:
- **Title** is usually the largest font, near top center of page 1
- **H1** has large bold font near top of a page
- **H2/H3** have decreasing font sizes and less emphasis
- Structural repetition and indentation aid disambiguation

Fallback to `pdfminer.six` ensures compatibility for basic PDFs.

## Output Format
Each output JSON has the following structure:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
