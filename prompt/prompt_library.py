from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

document_analysis_prompt = ChatPromptTemplate.from_template("""
you are a highly skilled and capable assistant. 
you have been trained to analyse and summarize documents.
                                          
{format_instructions}
                                          
Analyze this document:
{document_text}
                                          
                                          
""")

document_comparison_prompt = ChatPromptTemplate.from_template("""
you will get two PDFS. your task include the following:
1. compare the content of the two pdf.
2. check for the differences between the two content.
3. specify where the change occurs by indicating the page number, and line number.
4. if there are no changes please say 'NO CHANGES IDENTIFIED'.

Input documents: 
{combined_documents}

your response should follow the instructions:
{format_instructions}                                                                                                   
""")


contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )), 
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])                                                              

PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,   
}


