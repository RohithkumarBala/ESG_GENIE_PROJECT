'''
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from exa_py import Exa
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
st.set_page_config(page_title="ESG Genie", page_icon="🌱")

# 1. Initialize Clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Load Local Knowledge (PDF Database)
if os.path.exists("./chroma_db"):
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    st.error("Error: chroma_db folder not found. Please run ingest.py first!")

# 3. THE ROUTER (Decides Local vs Web)
def get_routing_decision(query):
    # Use the 'query' passed into the function, not final_prompt
    routing_prompt = f"Analyze: {query}. Reply LOCAL for internal docs or WEB for live news. One word only."
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", 
        contents=routing_prompt
    )
    return response.text.strip().upper()

# 4. STREAMLIT UI
st.title("🌱 ESG Genie")
st.markdown("### Hybrid Intelligence: Internal Docs + Live Web Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about ESG reports or latest market trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agentic Decision
    with st.spinner("🤖 Thinking..."):
        decision = get_routing_decision(prompt)
    
    source_tag = "🔍 (Internal Doc)" if "LOCAL" in decision else "🌐 (Live Web)"
    
    with st.chat_message("assistant"):
        with st.spinner(f"Querying {source_tag}..."):
            if "LOCAL" in decision:
                results = vector_db.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
            else:
                search_results = exa.search_and_contents(prompt, num_results=3, text=True)
                context = "\n".join([res.text for res in search_results.results])

            # Final RAG Synthesis
            final_prompt = f"Using this context: {context}\n\nQuestion: {prompt}\n\nResponse Style: EY Consultant."
            response = client.models.generate_content(model="gemini-2.0-flash-lite", contents=final_prompt)
            
            output = f"**Source:** {source_tag}\n\n{response.text}"
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
            '''
# Note: The above code is a simplified version of the ESG Genie chatbot application. It includes:
#- Initialization of Gemini and Exa clients
import streamlit as st
import os
import time
from dotenv import load_dotenv
from google import genai
from exa_py import Exa
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. INITIAL SETUP
load_dotenv()
st.set_page_config(page_title="ESG Genie", page_icon="🌱", layout="wide")

# Initialize API Clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

# Use the stable embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize Session State for Cache and Chat
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. LOAD DATABASE SAFELY
vector_db = None
if os.path.exists("./chroma_db"):
    try:
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        st.error(f"⚠️ Database Error: {e}")
else:
    st.sidebar.warning("📁 'chroma_db' not found. Run ingest.py to enable PDF search.")

# 3. HELPER FUNCTIONS
def get_routing_decision(query):
    """Determines if the question needs the PDF (LOCAL) or the Web (WEB)."""
    routing_prompt = f"Analyze ESG Query: {query}. Reply 'LOCAL' for internal doc stats or 'WEB' for news/trends. One word only."
    
    for _ in range(3): # Retry loop for 429 errors
        try:
            response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=routing_prompt)
            return response.text.strip().upper()
        except Exception as e:
            if "429" in str(e):
                time.sleep(5)
            else:
                break
    return "LOCAL" # Default fallback

# 4. STREAMLIT UI
st.title("🌱 ESG Genie")
st.caption("Strategic Sustainability Advisor | Hybrid RAG Intelligence")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about sustainability targets or market trends..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # CHECK CACHE FIRST (Saves your Quota!)
    if prompt in st.session_state.query_cache:
        cached_data = st.session_state.query_cache[prompt]
        output = f"⚡ **(Cached Response)**\n\n**Source:** {cached_data['source']}\n\n{cached_data['answer']}"
        with st.chat_message("assistant"):
            st.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
    
    else:
        # FULL RAG PROCESS
        with st.spinner("🤖 Routing & Retrieving..."):
            decision = get_routing_decision(prompt)
            source_tag = "🔍 Internal PDF" if "LOCAL" in decision else "🌐 Live Web"
            
            context = ""
            # Retrieve Data
            if "LOCAL" in decision and vector_db is not None:
                results = vector_db.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in results])
            else:
                try:
                    search = exa.search_and_contents(prompt, num_results=3, text=True)
                    context = "\n".join([r.text for r in search.results])
                except:
                    context = "Could not retrieve web data. Answering from general knowledge."

            # Final Synthesis with Retry
            final_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nRole: Act as an EY Sustainability Consultant. Provide a professional, data-driven answer."
            
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                try:
                    res = client.models.generate_content(model="gemini-2.5-flash-lite", contents=final_prompt)
                    clean_answer = res.text
                    
                    # Store in Cache
                    st.session_state.query_cache[prompt] = {
                        "answer": clean_answer,
                        "source": source_tag
                    }
                    
                    full_display = f"**Source:** {source_tag}\n\n{clean_answer}"
                    answer_placeholder.markdown(full_display)
                    st.session_state.messages.append({"role": "assistant", "content": full_display})
                
                except Exception as e:
                    if "429" in str(e):
                        st.error("🚦 Quota Exhausted. Please wait 1 minute before asking again.")
                    else:
                        st.error(f"❌ Error: {e}")