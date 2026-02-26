Milestone2 (
RAG stands for Retreival-Augmented Generation.
What is A RAG model?
A RAG is a Hybrid system that combines 2 key components:
1) Retreiver: this searches a knowledge base or database to identify the most relevant context for a qiven query.
2) Generator: the LLm that uses both the user prompt and the retrieved context to generate grounded answers.

RAG is a framework that allowas LLMS to generate content with real-time access to external knowledge, also 

RAG Architecture:
1) Load documents from the data source
2) split documents into smaller text chunks
3) convert chunks into embeddings
4) store the embeddings into vector database
5) Convert the user question into an embedding.
6) Retrieve the most similar document chunks .
7) Combine retrieved context with the user question in a prompt.
8) send the prompt to the LLM
9) LLM generates an answer using the provided context.
10) Return the final response to the user. 

