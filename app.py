import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
import openai
import os
import time
import extra_streamlit_components as stx
import datetime
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from PIL import Image
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from win10toast import ToastNotifier
import sqlite3
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

toaster = ToastNotifier()
# Read API key and parameters from file and set variables
with open('api_key.txt', 'r') as f:
    api_info = f.read().strip().split('\n')
    api_key = api_info[0].split('=')[1].strip()
    api_base = api_info[1].split('=')[1].strip()
    api_type = api_info[2].split('=')[1].strip()
    api_version = api_info[3].split('=')[1].strip()

# Set OpenAI API key and parameters
openai.api_key = api_key
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version

# Set deployment ID
DEPLOYMENT_ID = 'gpt-35-turbo'
#DEPLOYMENT_ID = 'text-davinci-003'


# Set the logo image path and height
logo_image_path = "images/eylogo1.png"
logo_height = 50
persist_directory = "chromastore"
model_name = "gpt-3.5-turbo"


# Load the EY logo image
logo_image = Image.open(logo_image_path)
logo_image = logo_image.resize((logo_height, logo_height))
st.set_page_config(page_title="EY Ask Me Bot", layout="wide", initial_sidebar_state="auto")
padding_top = 0
file_path = ""


st.title("EY Ask Me Bot")

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d")

# Append timestamp to the upload directory name
upload_dir = f"source_file"

# Set the OpenAI API key
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = api_key
#embeddings = OpenAIEmbeddings(deployment="embeddings",model="text-embedding-ada-002", chunk_size = 1)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
with st.sidebar:
    def add_logo(logo_path, width, height):
        """Read and return a resized logo"""
        logo = Image.open(logo_path)
        modified_logo = logo.resize((width, height))
        return modified_logo

    st.sidebar.image(add_logo(logo_path="images/eylogo1.png", width=50, height=60))
    st.sidebar.divider()

    selected = option_menu(
        menu_title=None,  # required
        options=["About & Guidelines", "Explore Options"],  # required
        icons=["info-circle-fill","chat-left-text"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        styles={
        "container": {"padding": "0!important"},
                "icon": {"color": "#ffffff", "font-size": "14px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#2e2e3c",
                },
        "nav-link-selected": {"background-color": "#3a3a4a",'color': "#ffe600"},
        }
    )

if selected == "About & Guidelines":
    st.write("About:")
    st.caption("EY Ask Me Bot, powered by OpenAI, offers three key functionalities to enhance your interaction. You can upload PDF documents to receive answers to your questions, query operations data from our database, and classify the Description column in a CSV file into five categories. With these features, our chat bot aims to provide convenient and efficient access to information, helping you make informed decisions and streamline your workflow.")
    st.write("Usage Guidelines:")
    st.write("Upload PDF:")
    st.caption("Upload your PDF document and ask questions related to its content.")
    st.caption("Our chat bot will use advanced natural language processing to provide accurate answers based on the uploaded PDF.")
    st.write("Operations Data:")
    st.caption("Enter specific keywords or queries to retrieve relevant information from operations database.")
    st.caption("Our chat bot will analyze your input and present the retrieved data in a user-friendly format.")
    st.write("DILO Data:")
    st.caption("Upload a CSV file with an ID and Description column.")
    st.caption("Our chat bot will classify the Description column into seven categories, helping you organize and analyze the data effectively.")
    st.write("Disclaimer:")
    st.caption("Please keep in mind that while our chat bot strives to offer reliable assistance, it's always advisable to review and validate the provided results for accuracy and suitability to your specific needs.")

if selected == "Explore Options":

    # Generate empty lists for generated and past.
    ## generated stores AI generated responses
    if 'generated_pdf' not in st.session_state:
        st.session_state['generated_pdf'] = ["I'm your Team's Assistant, How may I help you?"]
    ## past stores User's questions
    if 'past_pdf' not in st.session_state:
        st.session_state['past_pdf'] = ['Hi!']

    # def load_docs(directory):
    #     loader = DirectoryLoader(directory,glob="**/*.pdf")
    #     documents = loader.load()
    #     return documents
    #
    # def split_docs(documents):
    #     text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    #     docs = text_splitter.split_documents(documents)
    #     return docs


    # with st.spinner('Loading Document...'):
    #     time.sleep(5)
    # # documents = load_docs(upload_dir)

    # documents_with_content = [doc for doc in documents if doc.page_content is not None]
    # try:
    #     if not documents_with_content:
    #         st.error("No documents with content loaded!")
    #     else:
    #         pages_doc = split_docs(documents)
    #
    #         vectordb = Chroma.from_documents(pages_doc, embeddings, persist_directory="./chromadb")
    #         vectordb.persist()
    #         print(pages_doc)
    #         #print("Successfully persisted into chroma db!")
    #
    #         st.success(f"Document(s) loaded!")
    # except:
    #     st.error(f"Error in loading document")


    # Layout of input/response containers
    if 'file_path' in st.session_state:
        file_path = st.session_state['file_path']
    response_container = st.container()
    colored_header(label='', description='', color_name='yellow-80')
    input_container = st.container()


    # Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You:", value="", key="input")
        return input_text


    ## Applying the user input box
    with input_container:
        user_input = get_text()


    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    def generate_response(prompt):
        reply = ""
        if prompt:
            vectordb_new = Chroma(persist_directory="./chromadb",embedding_function=embeddings)
            print("vectordb_new = ",vectordb_new )
            print("persist_directory = ",persist_directory)
            print("prompt = ",prompt)

            llm_pdf = ChatOpenAI(engine=DEPLOYMENT_ID)
            chain = load_qa_chain(llm_pdf, chain_type="stuff")
            matching_docs = vectordb_new.similarity_search(prompt)
            print("matching_docs = ",matching_docs[0].page_content)

            reply = chain.run(input_documents=matching_docs, question=prompt)
            print("reply = ",reply)
        else:
            st.warning("Please enter a query.")
        return reply


    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            # try:
                response = generate_response(user_input)
                st.session_state.past_pdf.append(user_input)
                st.session_state.generated_pdf.append(response)
            # except:
                # st.error(
                #     "An error occurred while retrieving the generated responses. "
                #     "Please try again with a different prompt.")
        if st.session_state['generated_pdf']:
            for i in range(len(st.session_state['generated_pdf'])):
                message(st.session_state['past_pdf'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated_pdf"][i], key=str(i))


hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

##footer

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
height :5%;
background-color: #2e2e38;
color: #FFFFFF;
text-align: center;
}

</style>
<div class="footer">
<p style="font-size:9px;">Powered by <br> GDS Tax Transformation Team</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

