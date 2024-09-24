import os
import tempfile
import validators
import streamlit as st
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

class ContentProcessor:
    def __init__(self):
        load_dotenv()
        self.configure_environment()
        self.configure_streamlit()

    def configure_environment(self):
        os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
        os.environ['LANGCHAIN_TRACING_V2'] = "true"
        os.environ['LANGCHAIN_PROJECT'] = "LangChain: Process Content from Multiple Sources"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    def configure_streamlit(self):
        st.set_page_config(page_title="LangChain: Process Content from Multiple Sources", page_icon="🦜")
        st.title("🦜 LangChain: Process Content from Multiple Sources")

    def calculate_chunk_size(self, text_length: int, model_context_length: int) -> int:
        target_chunk_size = model_context_length // 3
        return max(1000, min(target_chunk_size, model_context_length // 2))

    def get_configuration(self) -> Dict[str, Any]:
        with st.sidebar:
            st.header("Configuration")
            groq_api_key = st.text_input("Groq API Key", type="password")
            model = st.selectbox("Select Model", ["llama3-8b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"])
            
            st.header("Task")
            task = st.radio("Choose task", ["Process Content", "Interactive Q&A"], index=0)
            
        return {"groq_api_key": groq_api_key, "model": model, "task": task}

    def get_sources(self) -> Dict[str, Any]:
        st.subheader('Select Sources to Process')
        use_urls = st.checkbox("URLs (YouTube or websites)")
        use_files = st.checkbox("File Upload (PDF or text files)")
        use_text = st.checkbox("Text Input")

        sources = {}
        if use_urls:
            sources['urls'] = st.text_area("Enter URLs (one per line)", placeholder="https://example.com\nhttps://youtube.com/watch?v=...")
        if use_files:
            sources['files'] = st.file_uploader("Upload PDF or text files", type=["pdf", "txt"], accept_multiple_files=True)
        if use_text:
            sources['text'] = st.text_area("Enter text content", placeholder="Paste your text here...")
        return sources

    def process_pdf(self, uploaded_file) -> List[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pdf_pages = loader.load()
        
        st.sidebar.write(f"Processing PDF: {uploaded_file.name}")
        st.sidebar.write(f"Total pages: {len(pdf_pages)}")

        os.unlink(temp_file_path)
        return pdf_pages

    def process_content(self, sources: Dict[str, Any]) -> List[Document]:
        all_docs = []

        if 'urls' in sources and sources['urls']:
            url_list = [url.strip() for url in sources['urls'].split('\n') if url.strip()]
            for url in url_list:
                if not validators.url(url):
                    st.warning(f"Skipping invalid URL: {url}")
                    continue
                
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    st.info(f"Processing YouTube video: {url}")
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    st.info(f"Processing website content: {url}")
                
                docs = loader.load()
                all_docs.extend(docs)

        if 'files' in sources and sources['files']:
            for uploaded_file in sources['files']:
                if uploaded_file.type == "application/pdf":
                    st.info(f"Processing PDF: {uploaded_file.name}")
                    all_docs.extend(self.process_pdf(uploaded_file))
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    loader = TextLoader(temp_file_path)
                    st.info(f"Processing text file: {uploaded_file.name}")
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.unlink(temp_file_path)

        if 'text' in sources and sources['text']:
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as temp_file:
                temp_file.write(sources['text'])
                temp_file_path = temp_file.name

            loader = TextLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
            st.info("Processing text input")
            os.unlink(temp_file_path)

        return all_docs

    def create_prompts(self) -> Dict[str, PromptTemplate]:
        prompt_template = """
        Provide a {action} of the following content:

        Content: {text}

        {action}:
        """

        refine_template = """
        We have provided an existing {action} of the content: {existing_answer}

        We have some additional content to incorporate: {text}

        Given this new information, please refine and update the existing {action}.

        Refined {action}:
        """

        return {
            "prompt": PromptTemplate(input_variables=['text', 'action'], template=prompt_template),
            "refine_prompt": PromptTemplate(input_variables=['text', 'action', 'existing_answer'], template=refine_template)
        }

    def process_documents(self, docs: List[Document], action: str, config: Dict[str, Any]) -> str:
        llm = ChatGroq(model=config['model'], groq_api_key=config['groq_api_key'])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.calculate_chunk_size(sum(len(doc.page_content) for doc in docs), 8192),
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        
        prompts = self.create_prompts()
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompts["prompt"],
            refine_prompt=prompts["refine_prompt"]
        )
        
        return chain.run(input_documents=split_docs, action=action.lower())

    def create_retriever(self, docs: List[Document]) -> FAISS:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)

    def answer_question(self, retriever: FAISS, question: str, config: Dict[str, Any]) -> str:
        llm = ChatGroq(model=config['model'], groq_api_key=config['groq_api_key'])
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever.as_retriever())
        return qa_chain.run(question)

    def run(self):
        config = self.get_configuration()
        sources = self.get_sources()
        
        if config['task'] == "Process Content":
            action_type = st.radio("Choose action type", ["Predefined", "Custom"])
            if action_type == "Predefined":
                action = st.selectbox("Select Action", self.predefined_actions)
            else:
                action = st.text_input("Enter Custom Action", placeholder="e.g., Summarize in bullet points")
        else:
            action = "Answer questions about the content"

        process_button = st.button("Process Content")
        
        if 'docs' not in st.session_state:
            st.session_state.docs = None
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None

        if process_button:
            if not config['groq_api_key'].strip():
                st.error("Please provide your Groq API Key in the sidebar.")
            elif not sources:
                st.error("Please select at least one source type and provide content.")
            elif config['task'] == "Process Content" and action_type == "Custom" and not action.strip():
                st.error("Please enter a custom action.")
            else:
                with st.spinner("Processing..."):
                    st.session_state.docs = self.process_content(sources)
                    
                    if not st.session_state.docs:
                        st.error("No content was processed. Please check your inputs and try again.")
                    elif config['task'] == "Process Content":
                        output = self.process_documents(st.session_state.docs, action, config)
                        st.success("Processing complete!")
                        st.subheader(f"{action} Result")
                        st.write(output)
                    else:  # Interactive Q&A
                        st.session_state.retriever = self.create_retriever(st.session_state.docs)
                        st.success("Document processed and ready for questions!")

        if config['task'] == "Interactive Q&A" and st.session_state.retriever is not None:
            question = st.text_input("Ask a question about the document:")
            if question:
                with st.spinner("Finding answer..."):
                    answer = self.answer_question(st.session_state.retriever, question, config)
                    st.subheader("Answer")
                    st.write(answer)

        st.divider()
        st.caption("Powered by LangChain and Groq")
        st.caption("Created by : Akshay Kumar BM")

    @property
    def predefined_actions(self):
        return [
            "Summarize", "Analyze", "Review", "Critique", "Explain",
            "Paraphrase", "Simplify", "Elaborate", "Extract key points",
            "Provide an overview", "Highlight main ideas", "Create an outline",
            "Generate a report", "Identify themes", "List pros and cons",
            "Fact-check", "Create study notes", "Generate questions"
        ]

if __name__ == "__main__":
    processor = ContentProcessor()
    processor.run()
