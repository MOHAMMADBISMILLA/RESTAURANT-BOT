from langchain_core.prompts import PromptTemplate

template = """
You are an information extraction assistant. Your job is to extract only the answer to the question below using the context provided.

- Do not explain your reasoning.
- Do not add any extra commentary.
- Only return the exact answer, nothing more.

If the answer is not found in the context, respond with: Not found.

make sure only give proper answer, not provide whole context

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)
