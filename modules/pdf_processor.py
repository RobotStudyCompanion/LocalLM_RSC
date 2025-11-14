import fitz  # PyMuPDF 
# need to install frontend package
import re
from typing import List, Dict, Tuple
from datetime import datetime
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
'''
function that will handle the Pdf converter
   Args:
       file_path: Path to the PDF file
       chunk_size: Target size for each text chunk (in characters)
       chunk_overlap: Number of overlapping characters between chunks
       min_chunk_size: Minimum chunk size to keep (filters out tiny fragments)
   
   Returns:
       Dictionary containing:
           - chunks: List of text chunks
           - metadata: List of metadata for each chunk
           - document_info: Overall document information
           - success: Boolean indicating success/failure
           - error: Error message if failed
this function will take a PDF file as input and will convert it to text/chunks for the storage :
- extract text from PDF using PyMuPDF (fitz)
- validate the file type (pdf)
- exctract the text, clean it and chu²nk it
- return structured data ready for embedding and storage

Intern functions used:

- clean_text : to clean the extracted text
- create_chunks : to create chunks from the cleaned text

 '''

def pdf_converter(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE
) -> Dict[str, any]:
    
    # Initialize result structure
    result = {
        "chunks": [],
        "metadata": [],
        "document_info": {},
        "success": False,
        "error": None
    }
    
    # ===== Step 1: Validation =====
    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result
    
    if not file_path.lower().endswith('.pdf'):
        result["error"] = f"File is not a PDF: {file_path}"
        return result
    
    try:
        # ===== Step 2: Open PDF and extract document info =====
        doc = fitz.open(file_path)
        
        result["document_info"] = {
            "filename": os.path.basename(file_path),
            "total_pages": len(doc),
            "processed_date": datetime.now().isoformat(),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }
        
        # ===== Step 3: Extract text page by page =====
        all_text = []
        page_boundaries = []  # Track which chunk belongs to which page
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Clean the text
            cleaned_text = clean_text(text)
            
            if cleaned_text.strip():  # Only add non-empty pages
                all_text.append({
                    "text": cleaned_text,
                    "page_num": page_num + 1  # 1-indexed for user-friendliness
                })
        
        doc.close()
        
        # ===== Step 4: Create chunks with overlap =====
        chunks = []
        metadata = []
        
        for page_data in all_text:
            page_text = page_data["text"]
            page_num = page_data["page_num"]
            
            # Split page into chunks
            page_chunks = create_chunks(
                page_text, 
                chunk_size, 
                chunk_overlap, 
                min_chunk_size
            )
            
            # Add metadata for each chunk
            for idx, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadata.append({
                    "page": page_num,
                    "chunk_index": len(chunks) - 1,
                    "chunk_size": len(chunk),
                    "page_chunk_index": idx,  # Index within the page
                    "source_file": result["document_info"]["filename"],
                    "processed_date": result["document_info"]["processed_date"]
                })
        
        result["chunks"] = chunks
        result["metadata"] = metadata
        result["success"] = True
        
        # Add statistics to document_info
        result["document_info"]["total_chunks"] = len(chunks)
        result["document_info"]["avg_chunk_size"] = (
            round(sum(len(c) for c in chunks) / len(chunks), 2) if chunks else 0
        )
        
    except Exception as e:
        result["error"] = f"Error processing PDF: {str(e)}"
        result["success"] = False
    
    return result


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text) # Remove excessive whitespace
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text) # Remove special characters but keep basic punctuation
    text = re.sub(r' +', ' ', text) # Remove multiple spaces
    text = text.strip() # Strip leading/trailing whitespace
    return text

def create_chunks(text: str, chunk_size: int, overlap: int, min_size: int) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size # Calculate end position
        # If this is not the last chunk, try to break at sentence boundary
        if end < text_length:
            # Look for sentence endings near the chunk boundary
            search_start = max(start, end - 100)  # Look back up to 100 chars
            search_text = text[search_start:end + 100]  # Look ahead up to 100 chars
            sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', search_text)]
            
            if sentence_ends:
                # Adjust end to the last sentence boundary
                last_sentence_end = sentence_ends[-1]
                end = search_start + last_sentence_end
        
        end = min(end, text_length)  # Ensure we don't exceed text length
        chunk = text[start:end].strip() # Extract chunk
        if len(chunk) >= min_size: # Only add chunks that meet minimum size
            chunks.append(chunk)
       
        next_start = end - overlap  # Move start position (with overlap)
        if next_start <= start: # Prevent infinite loop: ensure we always move forward
            next_start = end
        start = next_start
    
    return chunks


######### use this only to get info on waht is happening ##################################
def print_pdf_summary(result: Dict) -> None:
    if not result["success"]:
        print(f"[ERROR] Error: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("📄 PDF PROCESSING SUMMARY")
    print("="*60)
    
    doc_info = result["document_info"]
    print(f"\n📋 Document: {doc_info['filename']}")
    print(f"📏 File Size: {doc_info['file_size_mb']} MB")
    print(f"📖 Total Pages: {doc_info['total_pages']}")
    print(f"🔢 Total Chunks: {doc_info['total_chunks']}")
    print(f"📊 Avg Chunk Size: {doc_info['avg_chunk_size']} characters")
    print(f"⏰ Processed: {doc_info['processed_date']}")
    
    print("\n" + "="*60)
    print("📦 SAMPLE CHUNKS (First 3)")
    print("="*60)
    
    for i in range(min(3, len(result["chunks"]))):
        chunk = result["chunks"][i]
        meta = result["metadata"][i]
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        
        print(f"\n📌 Chunk {i+1} (Page {meta['page']}):")
        print(f"   Size: {meta['chunk_size']} chars")
        print(f"   Preview: {preview}")
    
    print("\n" + "="*60 + "\n")
