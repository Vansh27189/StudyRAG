import os
import re
import streamlit as st
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

st.set_page_config(page_title="RAGTube", layout="centered")

st.title("🎥 RAGTube")
st.info("👉 Please enter a YouTube video URL and then type your question before clicking **Ask**.")

url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


question =st.text_input(
    "Your Question",
    placeholder="What is this video about?"
)


def extract_video_id(url: str) -> str:
    short_match = re.search(r"youtube\.be/([^?&]+)",url)
    if short_match:
        return short_match.group(1)
    
    shorts_match = re.search(r"youtube\.com/shorts/([^?&]+)", url)
    if shorts_match:
        return shorts_match.group(1)
    
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    if "v" in query:
        return query["v"][0]


    raise ValueError("Invalid YouTube URL")


#ID EXTRACTOR FROM URL

if st.button("ASK"):
    if not url or not question:
        st.warning("Please enter both URL and question.")
    else:
        with st.spinner("Fetching transcript and generating answer..."):
            try:
                #STEP 1 - INDEXING (CALLING YT API)

                video_id = extract_video_id(url) #only ID
                try:
                    api = YouTubeTranscriptApi() #INSTANCE

                    # 1. List available transcript tracks
                    transcripts = api.list(video_id) #Transcript object - language, language_code , Fetch

                    
                    chosen = None
                    for t in transcripts:
                        if t.language_code == "en": # FROM transcript object - language code
                            chosen = t  #chosen → Transcript object(language="English", code="en")
                            break
                        if t.language_code == "hi":
                            chosen = t

                    if not chosen:
                        raise Exception("No Hindi or English captions available")

                    # 3. Fetch transcript
                    fetched = chosen.fetch()  


                #     FETCHED TRANSCRIPT OBJECT 
                #     fetched.snippets = [
                #     Snippet(text="Hello everyone", start=0.0, duration=1.5),
                #     Snippet(text="Today we will learn how artificial intelligence works", start=1.6, duration=4.0),
                #     Snippet(text="AI is changing the world very fast", start=5.7, duration=3.2)
                # ]
                    

                    # 4. Convert to raw format (list of dicts)
                    transcript_list = fetched.to_raw_data() # remove object and get inner data like text,start,duration from above fetched object

                    # 5. Join into plain text
                    transcript = " ".join(chunk["text"] for chunk in transcript_list)
                    # print(transcript)

                except Exception as e:
                    raise Exception(f"Transcript error: {e}")


                ## STEP 1B - TEXT SPLITTER
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript]) # MADE langchain DOCUMENT with PAGE_CONTENT AND METADATA of TEXT stored in transcript variable and split and made chunks

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
                    You are a helpful assistant.

                    The transcript context may be in any language.
                    ALWAYS answer in ENGLISH.

                    Answer only from the provided transcript context.
                    If the context is insufficient, say "I don't know".
                    context - {context}
                    question - {question}
                ''',
                input_variables=['context','question']
                )

                



                # CHAINS

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(joined_text),
                    'question': RunnablePassthrough()
                })
                parser = StrOutputParser()
                main_chain = RunnableSequence(parallel_chain | prompt | llm | parser)
                final_answer = main_chain.invoke(question)
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
                


