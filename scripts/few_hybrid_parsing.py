#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid HTML Parser using BeautifulSoup + Docling

Author: Daria
Goal: Extract and clean main text content from HTML news articles.
Implements a hybrid parsing strategy combining BeautifulSoup and Docling for robustness and quality.
"""

# Imports
from bs4 import BeautifulSoup
import pandas as pd
import os
import matplotlib.pyplot as plt
from docling.document_converter import DocumentConverter
import re

# Create a Docling converter
docling_converter = DocumentConverter()

def parse_with_bs4(html_content):
    """
    Parse HTML content using BeautifulSoup
    
    Args:
        html_content (str): HTML content as string
        
    Returns:
        dict: Dictionary containing titles, body text, and paragraphs
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Identify main article sections
    sections = soup.find_all('div', class_='text-image cq-dd-image')
    
    all_titles = []
    all_paragraphs = []

    footer_keywords = ['newsletter', 'staffnet', 'globe', 'download']

    for section in sections:
        # Preserve figcaption if it exists
        for fig in section.find_all('figure'):
            # Preserve figcaption if it exists
            figcaption = fig.find('figcaption')
            if figcaption:
                caption_text = figcaption.get_text(" ", strip=True)
                if caption_text:
                    all_paragraphs.append(caption_text)
            fig.decompose()

        # Skip footer sections based on title
        h2 = section.find('h2')
        if h2:
            title_text = h2.get_text(strip=True)
            if any(kw in title_text.lower() for kw in footer_keywords):
                continue
            all_titles.append(title_text)
        else:
            # If no h2, add "Main article" title for first/main content
            if not all_titles:
                all_titles.append("Main article")

        #  Paragraph extraction
        for p in section.find_all('p'):
            text = p.get_text(" ", strip=True)

            # Clean up link artifacts like "external page", "call_made"
            text = text.replace("external page", "").replace("call_made", "")
            text = ' '.join(text.split())  # Normalize extra spaces
            
            if not text:
                continue
            if any(kw in text.lower() for kw in footer_keywords):
                continue
            if len(text) < 20:
                continue
            all_paragraphs.append(text)
            
    return {
        'titles': all_titles,
        'body': '<br><br>'.join(all_paragraphs),  # use <br><br> for clearer formatting
        'paragraphs': all_paragraphs
    }

def parse_with_docling(filepath):
    """
    Parse HTML file using Docling
    
    Args:
        filepath (str): Path to the HTML file
        
    Returns:
        dict: Dictionary containing body text in markdown format
    """
    result = docling_converter.convert(filepath)
    text = result.document.export_to_markdown()
    return {
        'body': text
    }

def inject_docling_bullets(bs4_body, docling_markdown):
    """
    Extract and inject bullet points from Docling markdown into BS4 body
    
    Args:
        bs4_body (str): Body text from BeautifulSoup
        docling_markdown (str): Markdown text from Docling
        
    Returns:
        str: Updated body text with bullet points
    """
    # Extract bullet-style lines from docling
    bullet_lines = [
        line.strip()
        for line in docling_markdown.splitlines()
        if line.strip().startswith(('•', '- '))
    ]

    print(" Bullet lines extracted from Docling:", bullet_lines)  # DEBUG

    # Keep only bullets not already in bs4_body
    unique_bullets = [
        bullet for bullet in bullet_lines
        if bullet not in bs4_body
    ]

    if not unique_bullets:
        return bs4_body

    # Prepend them at the top with <br> spacing
    bullet_block = '<br><br>' + '<br><br>'.join(unique_bullets) + '<br><br>'
    return bullet_block + bs4_body

def hybrid_parser_from_content(filename, html_content):
    """
    Hybrid parser that works with in-memory HTML content
    
    Args:
        filename (str): Filename for the HTML content
        html_content (str): HTML content as string
        
    Returns:
        dict: Dictionary containing parsed data from both parsers
    """
    # Use BeautifulSoup to extract clean body and titles
    bs4_data = parse_with_bs4(html_content)

    # Save HTML to a temporary file to pass to Docling
    temp_path = f"/tmp/{filename}"
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Use Docling to extract structured markdown
    docling_data = parse_with_docling(temp_path)

    return {
        'filename': filename,
        'bs4_body': bs4_data['body'],
        'bs4_titles': bs4_data['titles'],
        'bs4_paragraphs': bs4_data['paragraphs'],
        'docling_markdown': docling_data['body']
    }

def chunk_docling_markdown(markdown_text):
    """
    Split docling markdown into sections using ## headers.
    Fallback to one generic chunk if headers are missing or irrelevant.
    
    Args:
        markdown_text (str): Markdown text from Docling
        
    Returns:
        list: List of dictionaries containing title and text
    """
    chunks = []
    current = {"title": None, "text": ""}
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current["title"] or current["text"].strip():
                chunks.append(current)
            current = {"title": line[3:].strip(), "text": ""}
        else:
            current["text"] += line + "\n"
    if current["title"] or current["text"].strip():
        chunks.append(current)

    # If only footers are present, ignore them and use a fallback chunk
    footer_keywords = ["staffnet", "newsletter", "kontakt", "about"]
    only_footers = all(any(kw in (c["title"] or "").lower() for kw in footer_keywords) for c in chunks)

    if len(chunks) < 2 or only_footers:
        return [{"title": "Main article", "text": ""}]
    else:
        return chunks

def distribute_bs4_text(paragraphs, docling_chunks):
    """
    Distribute list of BS4 paragraphs into Docling-defined chunks
    
    Args:
        paragraphs (list): List of paragraphs from BeautifulSoup
        docling_chunks (list): List of chunks from Docling
        
    Returns:
        list: Updated list of chunks with body text
    """
    n = len(docling_chunks)
    m = len(paragraphs)
    avg = max(1, m // n) if n > 0 else m

    for i, chunk in enumerate(docling_chunks):
        start = i * avg
        end = m if i == n - 1 else (i + 1) * avg
        chunk["body"] = "\n\n" + "\n\n".join(paragraphs[start:end])
    
    return docling_chunks

def main():
    """
    Main function to process example HTML files
    """
    # Read example files
    example_files = [
        'HKNews/de_internal/2015/05/die-eth-karte-erhaelt-ein-neues-design.html',
        'HKNews/de_news_events/2016/04/erc-advanced-grants.html',
        'HKNews/en_internal/2020/08/in-memory-of-konrad-steffen.html',
        'HKNews/en_news_events/2024/03/detecting-storms-thanks-to-gps.html'
    ]

    examples = {}
    for filepath in example_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                examples[os.path.basename(filepath)] = f.read()
        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
            
    if not examples:
        print("No example files found. Please update the file paths.")
        return

    # Display example names
    print("Processing files:", list(examples.keys()))

    # Apply hybrid parser to examples
    hybrid_results = []
    for filename, html_content in examples.items():
        result = hybrid_parser_from_content(filename, html_content)
        hybrid_results.append(result)

    # Create final chunks
    final_chunks_per_file = []
    for document in hybrid_results:
        # Step 1: Create structural chunks from the markdown document
        docling_chunks = chunk_docling_markdown(document['docling_markdown'])
        
        # Step 2: Distribute the BS4 text content into the markdown structure
        content_chunks = distribute_bs4_text(document['bs4_paragraphs'], docling_chunks)

        # Step 3: Inject bullet points into the first chunk's body (only if needed)
        if content_chunks:
            content_chunks[0]["body"] = inject_docling_bullets(
                bs4_body=content_chunks[0]["body"],
                docling_markdown=document['docling_markdown']
            )
        
        # Add the processed document to results
        final_chunks_per_file.append({
            'filename': document['filename'],
            'chunks': content_chunks
        })

    # Preview results
    for f in final_chunks_per_file:
        print(f"\n=== {f['filename']} ===")
        for c in f['chunks']:
            print(f"\n## {c['title']}\n{c['body']}")

    # Write results to files
    output_dir = "parsed_markdown"
    os.makedirs(output_dir, exist_ok=True)

    for f in final_chunks_per_file:
        filename = f"{f['filename'].replace('.html', '')}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(f"# {f['filename']}\n\n")
            for c in f['chunks']:
                out.write(f"## {c['title']}\n\n")
                paragraphs = c['body'].split('<br><br>')  
                for p in paragraphs:
                    clean = p.strip()
                    if clean:
                        out.write(clean + "\n\n")  # Force paragraph block

if __name__ == "__main__":
    main()