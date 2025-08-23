from langchain_core.prompts import ChatPromptTemplate

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


PROMPT_REGISTRY = {"document_analysis":document_analysis_prompt,"document_comparison": document_comparison_prompt }


