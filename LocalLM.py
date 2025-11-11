"""
Three-Tier LLM 

Architecture :

Three  Tier :
- first tier compose of caching that will handle response to question already in the database or almost similar
- Second tier that will handle question that are similar but not enough to be handle by the cache :
    Local LM  ( 1B model )
- Third tier that will handle question that are not similar enough to be handle by the cache or the local LM :
    Local LM ( 4B model )

So as Models we will use :
- One Local LM that is gonna be around 4B (gemma, mistral, etc)
- One Local LM that is gonna be around 1B or lower (Gemma 3 )
- One Embedding Model that is gonna be around 1B or lower (Gemma Embedding Model )

Tools used :
- memory Vector database (TBD)
- ollama  for local LLMs
- PDF loader for document ingestion
- TTS
- STT

Functionality : 
- Document Ingestion from PDF
- Question Generation from the document
- storage of the document embedding in the vector database
- storage of the generated questions and answers in the vector database
- storage of the user questions and answers in the vector database
- accuracy evalutation of the user questions compared to those in the database
- Handeling of different scenarios based on the accuracy evaluation :
    - if the accuracy is high enough, return the answer from the database
    - if the accuracy is medium, use the 1B local LM to answer the question
    - if the accuracy is low, use the 4B local LM to answer the question
- TTS of the final answer

Description of the use:

This code is designed to answers the need of a fast and accurate answering from user questions. 
This architecture will be adaptable to any subject and will be able to run on a Raspberry Pi 4 with 8GB of RAM.

Libraries used :
-ollama : to handle local LLMs
-nmupy 
-datetime

"""
import ollama
import numpy as np
from datetime import datetime
from pdf_processor import pdf_converter






###############################################################################################
# function that will Store in the database
###############################################################################################
###############################################################################################
# function that will handle accuracy evaluation
###############################################################################################
###############################################################################################
# function that will handle the three tier architecture
###############################################################################################
###############################################################################################
# function that will handle the save of the new questions from the users in the database
###############################################################################################
###############################################################################################
# function that will handle the generation of questions from the document using the big model
###############################################################################################
###############################################################################################
# function that will handle the second tier LLM (1B model) that will need to answers based on the similar questions and the similar found in the documents
###############################################################################################



if __name__ == "__main__":
    pass