import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os 
import tempfile
#

st.set_page_config(page_title="LangChain: Process Content from Multiple Sources", page_icon="🦜")
st.title("🦜 LangChain: Process Content from Multiple Sources")

if 'pdf_page_ranges' not in st.session_state:
    st.session_state.pdf_page_ranges = {}

def calculate_chunk_size(text_length, model_context_length):
    target_chunk_size = model_context_length // 3
    return max(1000, min(target_chunk_size, model_context_length // 2))

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    model = st.selectbox("Select Model", ["llama3-8b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"])
    
    st.header("PDF Settings")

st.subheader('Select Sources to Process')
use_urls = st.checkbox("URLs (YouTube or websites)")
use_files = st.checkbox("File Upload (PDF or text files)")
use_text = st.checkbox("Text Input")

sources = {}

if use_urls:
    sources['urls'] = st.text_area("Enter URLs (one per line)", placeholder="https://example.com\nhttps://youtube.com/watch?v=...")

if use_files:
    uploaded_files = st.file_uploader("Upload PDF or text files", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        sources['files'] = uploaded_files
        
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                loader = PyPDFLoader(temp_file_path)
                pdf_pages = loader.load()
                total_pages = len(pdf_pages)

                file_key = f"pdf_range_{uploaded_file.name}"
                
                if file_key not in st.session_state.pdf_page_ranges:
                    st.session_state.pdf_page_ranges[file_key] = (1, total_pages)

                with st.sidebar:
                    st.write(f"PDF: {uploaded_file.name}")
                    st.write(f"Total pages: {total_pages}")
                    if total_pages > 1:
                        page_range = st.slider(
                            f"Select page range for {uploaded_file.name}",
                            1, total_pages, 
                            value=st.session_state.pdf_page_ranges[file_key],
                            key=file_key
                        )
                        st.session_state.pdf_page_ranges[file_key] = page_range
                    else:
                        st.write("This PDF has only one page.")
                        st.session_state.pdf_page_ranges[file_key] = (1, 1)

                os.unlink(temp_file_path)

if use_text:
    sources['text'] = st.text_area("Enter text content", placeholder="Paste your text here...")

predefined_actions = [
    "Summarize", "Analyze", "Review", "Critique", "Explain",
    "Paraphrase", "Simplify", "Elaborate", "Extract key points",
    "Provide an overview", "Highlight main ideas", "Create an outline",
    "Generate a report", "Identify themes", "List pros and cons",
    "Fact-check", "Create study notes", "Generate questions"
]

action_type = st.radio("Choose action type", ["Predefined", "Custom"])

if action_type == "Predefined":
    action = st.selectbox("Select Action", predefined_actions)
else:
    action = st.text_input("Enter Custom Action", placeholder="e.g., Summarize in bullet points")

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

prompt = PromptTemplate(input_variables=['text', 'action'], template=prompt_template)
refine_prompt = PromptTemplate(input_variables=['text', 'action', 'existing_answer'], template=refine_template)

if st.button("Process Content"):
    if not groq_api_key.strip():
        st.error("Please provide your Groq API key in the sidebar.")
    elif not sources:
        st.error("Please select at least one source type and provide content.")
    elif action_type == "Custom" and not action.strip():
        st.error("Please enter a custom action.")
    else:
        try:
            llm = ChatGroq(model=model, groq_api_key=groq_api_key)
            
            all_docs = []
            
            with st.spinner(f"Processing... ({action.lower()})"):
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
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_file_path = temp_file.name

                        if uploaded_file.type == "application/pdf":
                            loader = PyPDFLoader(temp_file_path)
                            st.info(f"Processing PDF: {uploaded_file.name}")
                            
                            pdf_pages = loader.load()
                            file_key = f"pdf_range_{uploaded_file.name}"
                            page_range = st.session_state.pdf_page_ranges[file_key]
                            
                            selected_pages = pdf_pages[page_range[0]-1:page_range[1]]
                            
                            chunk_size = calculate_chunk_size(sum(len(page.page_content) for page in selected_pages), 8192)
                            current_chunk = []
                            current_chunk_size = 0
                            
                            for page in selected_pages:
                                page_size = len(page.page_content)
                                if current_chunk_size + page_size > chunk_size and current_chunk:
                                    all_docs.append(Document(page_content="\n".join([p.page_content for p in current_chunk]), metadata={"source": uploaded_file.name}))
                                    current_chunk = []
                                    current_chunk_size = 0
                                current_chunk.append(page)
                                current_chunk_size += page_size
                            
                            if current_chunk:
                                all_docs.append(Document(page_content="\n".join([p.page_content for p in current_chunk]), metadata={"source": uploaded_file.name}))
                        else:
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
                
                if not all_docs:
                    st.error("No content was processed. Please check your inputs and try again.")
                    
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=calculate_chunk_size(sum(len(doc.page_content) for doc in all_docs), 8192), chunk_overlap=200)
                split_docs = []
                for doc in all_docs:
                    if doc.metadata.get("source", "").lower().endswith(".pdf"):
                        split_docs.append(doc)
                    else:
                        split_docs.extend(text_splitter.split_documents([doc]))
                
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="refine",
                    question_prompt=prompt,
                    refine_prompt=refine_prompt
                )
                
                output = chain.run(input_documents=split_docs, action=action.lower())
                
                st.success("Processing complete!")
                st.subheader(f"{action} Result")
                st.write(output)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.divider()
st.caption("Powered by LangChain and Groq")
