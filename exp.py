from retrieval import RetrievalSystem

from app.rag import RAGPipeline

from langchain.docstore.document import Document

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####" , "Header4"),
    ("#####","header5") ,
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)




with open('output_cleaned.md', 'r', encoding='utf-8') as file:
    research_paper_markdown = file.read()

docs_by_section = markdown_splitter.split_text(research_paper_markdown)



char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    )


final_chunks = []

for doc in docs_by_section:

    chunks = char_splitter.split_text(doc.page_content)
    for chunk in chunks:
        final_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

for idx, chunk in enumerate(final_chunks):
    print(f"Chunk {idx + 1}:\n{chunk.page_content}\nMetadata: {chunk.metadata}\n")


retrieval_system = RetrievalSystem()
retrieval_system.build_vector_store(final_chunks)  # Your document chunks
rag_pipeline = RAGPipeline(retrieval_system)
