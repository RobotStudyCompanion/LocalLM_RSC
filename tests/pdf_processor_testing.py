from pdf_processor import *
import os

def main():
    
    # 2. Process PDFs
    print("📄 Processing documents...")
    pdf_path = "C:/Users/doria/Desktop/LocalLM_RSC/pdfcache.pdf"
    result = pdf_converter(pdf_path)
    
    if result["success"]:
        print(f"✅ Processed {len(result['chunks'])} chunks")
        print_pdf_summary(result)
    else:
        print(f"❌ Error processing PDF: {result['error']}")
        


if __name__ == "__main__":
    main()