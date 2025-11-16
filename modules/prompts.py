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
You are a helpful study companion that assists students with their questions.
You should always answer in a concise and clear manner. You should prioritize accuracy in your answers.
You should use simple and easy to understand language, suitable for a student audience.
You should make the answer engaging and interesting.
You should use language as if you were talking orally to a student.

IMPORTANT INSTRUCTIONS:
1. First, check if the question is related to the provided context
2. If the question IS related to the context, use the context information to answer
3. If the question is NOT related to the context (different topic entirely), IGNORE the context completely and answer the question directly from your own knowledge
4. DO NOT try to force connections between unrelated topics
5. Answer questions on any safe topic, whether or not it relates to the context

Here is some context that MAY or MAY NOT be relevant:
{context_text}

Question: {query}

Answer (use context only if relevant, otherwise answer directly):"""

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


def get_llm_prompt_general_question(query: str) -> str:
    """
    Prompt for answering general questions that are not related to any document context.
    Used when similarity is too low to warrant using context.

    Args:
        query: User's question

    Returns:
        Formatted prompt string
    """
    return f"""
You are a helpful study companion that assists students with their questions.
You should always answer in a concise and clear manner. You should prioritize accuracy in your answers.
You should use simple and easy to understand language, suitable for a student audience.
You should make the answer engaging and interesting.
You should use language as if you were talking orally to a student.

Answer the following question directly and accurately:

Question: {query}

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
    return f"""you are a student trying to learn the material provided in the text below, by generating possible questions and answers that could be asked about the material.
Those questions should be created like they were asked orally by another student that didn't understand the material well enough.

Based on the following text, generate {num_questions} diverse questions that can be answered using the information in the text. For each question, provide a clear and concise answer.

CRITICAL REQUIREMENTS FOR QUESTIONS:
- Each question MUST be 10 words or less (STRICT LIMIT - count the words!)
- Questions should be 3-10 words long
- Questions should sound natural and conversational
- Questions should cover different aspects of the text
- DO NOT exceed 10 words per question under any circumstances

Text:
{chunk}

Format your response EXACTLY as:
Q1: [question - maximum 10 words]
A1: [answer]

Q2: [question - maximum 10 words]
A2: [answer]

Q3: [question - maximum 10 words]
A3: [answer]

Generate the questions and answers now. Remember: each question MUST be 10 words or less!"""

