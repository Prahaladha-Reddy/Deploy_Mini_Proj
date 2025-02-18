from typing import Dict, List, Any
from langchain.chains.base import Chain
from retrieval import RetrievalSystem
from chunks import chunks_maker
from retrieval import RetrievalSystem
from rag import RAGPipeline
from typing import Dict, List, Any
from langchain.chains.base import Chain
import re
import base64
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

chunks=chunks_maker(path="output_cleaned.md")
chunks.makedown_splitter()
final_chunks=chunks.recursive_character_splitter()


retriever=RetrievalSystem(embeddings)
retriever.build_vector_store(final_chunks)
rag=RAGPipeline(retriever)

llm=ChatGroq()
class DocumentRetrievalChain(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return ["documents"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        docs = retriever.retrieve(query)
        return {"documents": docs}
    


class ImageRetrievalChain(Chain):

    @property
    def input_keys(self) -> List[str]:
        return ["documents"]
        
    @property
    def output_keys(self) -> List[str]:
        return ["images"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = inputs["documents"]
        figg = [
            page
            for page in docs
            if "Figure" in page.page_content or "Table" in page.page_content
        ]
        docu = ""
        for i in figg:
            page_conn = i.page_content
            docu += page_conn
        patternn = r"\b(?:Figure|Table)\s\d+\b"
        matches_ptrn = re.findall(patternn, docu)
        bcd = set()
        matchess = []
        for amatch in matches_ptrn:
            if amatch not in bcd:
                matchess.append(amatch)
                bcd.add(amatch)
        image_references = list(
            map(lambda x: "extracted_data_fugg/" + x + ".jpg", matchess)
        )

        return {"images": image_references}



class Encode_Images(Chain):
    @property
    def input_keys(self) -> List[str]:
      """Input keys."""
      return ["documents","images", "query"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        return ["Encoded_Images"]

    def _call(self, inputs):
        docs = inputs["documents"]
        query = inputs["query"]
        images=inputs['images']

        Encoded_Images=[]
        for path in images:
          with open(path, "rb") as img_file:
              base64_str = base64.b64encode(img_file.read()).decode('utf-8')
              Encoded_Images.append(base64_str)
        return {"Encoded_Images": Encoded_Images}
  
class LLMChain(Chain):

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["documents","Encoded_Images", "query"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        return ["response"]

    def _call(self, inputs):
        docs = inputs["documents"]
        query = inputs["query"]
        Encodes_images=inputs['Encoded_Images']
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an excellent professor of AI, often compared to the greats like Feynman.
        explain every little detail that you find in context which is relevant to the question
        Use the following context to answer the question:
        Context:
        {context}

        Question:
        {query}
        """
        messages = [HumanMessage(content=prompt)]
        response = llm(messages)

        return {"response": (response.content,Encodes_images,query)}





