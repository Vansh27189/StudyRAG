import os
import re
import streamlit as st
import pypdf
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []




st.set_page_config(page_title="StudyRAG", layout="centered")

st.title("🎥 StudyRAG")
st.info(
    "Upload your lecture notes (PDF or TXT) and ask questions. "
    "Answers are generated only from your notes."
)

uploaded_file = st.file_uploader(
    "UPLOAD YOUR NOTES",
    type=["pdf","txt"]
)


question =st.text_input(
    "Your Question",
    placeholder="What is backpropagation?"
)

def read_pdf(file):
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file):
    return file.read().decode("utf-8")




#ID EXTRACTOR FROM URL



if st.button("ASK"):
    if not uploaded_file or not question:
        st.warning("Please upload notes and enter a question.")
    else:
        with st.spinner("Processing notes and generating answer..."):
            try:
                if uploaded_file.type == "application/pdf":
                    text = read_pdf(uploaded_file)
                else:
                    text = read_txt(uploaded_file)


                ## STEP 1B - TEXT SPLITTER
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([text]) # MADE langchain DOCUMENT with PAGE_CONTENT AND METADATA of TEXT stored in transcript variable and split and made chunks

                ##EMBEDDING 
                embeddings = HuggingFaceEndpointEmbeddings(
                    model = "sentence-transformers/all-MiniLM-L6-v2",
                    task="feature-extraction",
                    huggingfacehub_api_token=os.getenv("HF_TOKEN")
                )
                vector_store = FAISS.from_documents(chunks, embeddings)
                # print(vector_store.index_to_docstore_id) #IT  PRINTS VECTOR EMBEDDINGS


                ## retriever

                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})

                def joined_text(docs):
                    joined_docs = "\n\n".join(d.page_content for d in docs)
                    return joined_docs


                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature = 0.2
                )

                prompt = PromptTemplate(
                    template=
                    ''' 
                    You are a careful document reader.  
                    Use ONLY the provided document context and chat history.  
                    Do not use outside knowledge.  
                    If the answer is not present, say "I don't know."  
                    
                    Chat History: {chat_history}  
                    Document Context: {context}  
                    Current Question: {question}


                ''',
                input_variables=['chat_history','context','question']
                )

                def format_history(history):
                    return "\n".join(
                    f"User: {q}\nAssistant: {a}"
                    for q, a in history
    )
                def get_chat_history(_):
                    return format_history(st.session_state.get("chat_history", []))



                # CHAINS

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(joined_text),
                    'question': RunnablePassthrough(),
                    'chat_history':RunnableLambda(get_chat_history)
                })
                parser = StrOutputParser()
                main_chain = RunnableSequence(parallel_chain | prompt | llm | parser)
                final_answer = main_chain.invoke(question)
                st.session_state.chat_history.append((question, final_answer))

                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            padding: 1.2rem;
                            border-radius: 10px;
                            background-color: #0f3d2e;
                            color: #eafff5;
                            font-size: 1.05rem;
                            line-height: 1.6;
                        ">
                        {final_answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                st.error((e))
                

if st.button("Clear Conversation"):
    st.session_state.chat_history = []
