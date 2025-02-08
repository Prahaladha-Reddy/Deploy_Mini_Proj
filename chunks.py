from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

class chunks_maker:
  def __init__(self,path) -> str:
    self.path_to_md=path

  def makedown_splitter(self):

    headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####" , "Header4"),
    ("#####","header5") ,
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)

    with open(self.path_to_md, 'r', encoding='utf-8') as file:
      research_paper_markdown = file.read()

    self.docs_by_section = markdown_splitter.split_text(research_paper_markdown)

  def recursive_character_splitter(self):
    char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    )
    self.final_chunks = []
    for doc in self.docs_by_section:
        chunks = char_splitter.split_text(doc.page_content)
        for chunk in chunks:
            self.final_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return self.final_chunks

      