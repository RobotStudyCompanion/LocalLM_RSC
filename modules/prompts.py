"""
Prompts Configuration File
All LLM prompts are centralized here for easy modification
"""

# ===== LLM HANDLER PROMPTS for the small lm =====

def get_llm_prompt_with_context_tier_2(query: str, context_text: str) -> str:
    """
    Prompt for answering queries with context (Tier 2 small model)

    Args:
        query: User's question
        context_text: Formatted context from database and similarity cache

    Returns:
        Formatted prompt string
    """
    return f"""
You are a robot study companion, you help students to answer questions based on provided context.
You should always answer in a concise and clear manner. You should preoritize accuracy in your answers.
You should use a simple and easy to understand language, suitable for a student audience on the specific topic.
You should try to make the answer engaging and interesting.
You should use language and way of explaining as your were talking by oral to a student.

Answer the question based on the provided contextn this context is from both the document database and a cache of previously answered questions.
Your goal is to use the information in the context to provide an accurate and concise answer, you should prioritize information from the context.

here is the context:
{context_text}

Question: {query}

Answer:"""

def get_llm_prompt_with_context_tier_3(query: str, context_text: str) -> str:
    """
    Prompt for answering queries with context (Tier 3, big model)

    Args:
        query: User's question
        context_text: Formatted context from database

    Returns:
        Formatted prompt string
    """
    return f"""
You are a robot study companion, you help students to answer questions based on provided context.
You should always answer in a concise and clear manner. You should preoritize accuracy in your answers.
You should use a simple and easy to understand language, suitable for a student audience on the specific topic.
You should try to make the answer engaging and interesting.
You should use language and way of explaining as your were talking by oral to a student.

Answer the question based on the provided contextn this context is from both the document database.
Your goal is to use the information in the context to provide an accurate and concise answer, you should prioritize information from the context if possible.

here is the context:
{context_text}

Question: {query}

Answer:"""

def get_llm_prompt_with_context(query: str, context_text: str) -> str:
    """
    Prompt for answering queries with context (Tier 2 and 3)

    Args:
        query: User's question
        context_text: Formatted context from database

    Returns:
        Formatted prompt string
    """
    return f"""You are a helpful assistant. Answer the question based on the provided context.

{context_text}

Question: {query}

Answer:"""


def get_llm_prompt_without_context(query: str) -> str:
    """
    Prompt for answering queries without context

    Args:
        query: User's question

    Returns:
        Formatted prompt string
    """
    return f"""Question: {query}

Answer:"""


# ===== QUESTION GENERATOR PROMPTS =====

def get_question_generation_prompt(chunk: str, num_questions: int = 3) -> str:
    """
    Prompt for generating Q&A pairs from document chunks

    Args:
        chunk: Text chunk from document
        num_questions: Number of questions to generate

    Returns:
        Formatted prompt string
    """
    return f"""Based on the following text, generate {num_questions} diverse questions that can be answered using the information in the text. For each question, provide a clear and concise answer.

Text:
{chunk}

Format your response as:
Q1: [question]
A1: [answer]

Q2: [question]
A2: [answer]

Q3: [question]
A3: [answer]

Generate the questions and answers now:"""

