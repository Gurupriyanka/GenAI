import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
import openai
import os
from textblob import TextBlob
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
# Read API key and parameters from file
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
DEPLOYMENT_ID = 'text-davinci-003'


# Set the logo image path and height
logo_image_path = "images/logo.png"
logo_height = 50
persist_directory = "chromastore"
model_name = "gpt-3.5-turbo"

# Load the logo image
logo_image = Image.open(logo_image_path)
logo_image = logo_image.resize((logo_height, logo_height))
st.set_page_config(page_title="AI Bot", layout="wide", initial_sidebar_state="auto")
padding_top = 0
selected_tab = ""
file_path = ""

st.title("GEN AI Bot")

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d")

# Append timestamp to the upload directory name
upload_dir = f"Web Uploaded_{timestamp}"

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


    st.sidebar.image(add_logo(logo_path="images/logo.png", width=50, height=60))
    st.sidebar.divider()

    selected = option_menu(
        menu_title=None,  # required
        options=["About & Guidelines", "Explore Options"],  # required
        icons=["info-circle-fill", "chat-left-text"],  # optional
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
            "nav-link-selected": {"background-color": "#3a3a4a", 'color': "#ffe600"},
        }
    )

if selected == "About & Guidelines":
    st.write("About:")
    st.caption(
        "AI Ask Me Bot, powered by OpenAI, offers three key functionalities to enhance your interaction. You can upload PDF documents to receive answers to your questions, query operations data from our database, and classify the Description column in a CSV file into five categories. With these features, our chat bot aims to provide convenient and efficient access to information, helping you make informed decisions and streamline your workflow.")
    st.write("Usage Guidelines:")
    st.write("PDF Summarization:")
    st.caption("Upload your PDF document and ask questions related to its content.")
    st.caption(
        "Our chat bot will use advanced natural language processing to provide accurate answers based on the uploaded PDF.")
    st.write("Ops Chatbot:")
    st.caption("Enter specific keywords or queries to retrieve relevant information from operations database.")
    st.caption("Our chat bot will analyze your input and present the retrieved data in a user-friendly format.")
    st.write("DILO Analysis:")
    st.caption("Upload a CSV file with an ID and Description column.")
    st.caption(
        "Our chat bot will classify the Description column into seven categories, helping you organize and analyze the data effectively.")
    st.write("Sentiment Analysis:")
    st.caption("Upload a CSV file with Feedback Id, Team Name and Comments column.")
    st.caption(
        "Our chat bot will analyze the sentiment of Comments column helping you getting Areas of Improvement and Areas of Delight column with sentiment score.")
    st.write("Disclaimer:")
    st.caption(
        "Please keep in mind that while our chat bot strives to offer reliable assistance, it's always advisable to review and validate the provided results for accuracy and suitability to your specific needs.")

if selected == "Explore Options":

    chosen_id = "default"

    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="tab_pdf", title="PDF Summarization", description="Get Answers from PDF"),
        stx.TabBarItemData(id="tab_util", title="Ops Chatbot", description="Get Answers for Employee Data"),
        stx.TabBarItemData(id="tab_dilo", title="DILO Analysis", description="Get Details of DILO data"),
        stx.TabBarItemData(id="tab_sentiment", title="Sentiment Analysis",
                           description="Analyse Sentiment of Comments")])

    placeholder = st.container()
    # Define categories
    categories = {
        '1': 'Client or Authorized codes',
        '2': 'Suspense codes',
        '3': 'Learning and development',
        '4': 'Recruitment, Induction and Onboarding',
        '5': 'Meeting and Team activities',
        '6': 'Internal Project Automation',
        '7': 'Idle time, Break time or leaves'
    }


    def insert_data_into_db(df):
        # Truncate table
        c.execute('DELETE FROM dilo_desc_class')
        # Insert data
        c.executemany('INSERT INTO dilo_desc_class (description, ID, category) VALUES (?, ?, ?)', df.values.tolist())
        conn.commit()


    # Function to download data from SQLite database
    def download_data_from_db():
        query = "SELECT * FROM dilo_desc_class"
        df = pd.read_sql_query(query, conn)
        return df


    # Function to classify description
    def classify_description(description):
        prompt = f"Classify the following description into the appropriate category:\n\nDescription: {description}\nCategories = {categories}\n\nCategory:"
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=1,
            n=1,
            temperature=0.0,
            stop=None
        )
        category_number = response.choices[0].text.strip()
        category = categories.get(category_number, 'Unknown')
        return category


    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        print("Loading and reading PDF..")
        pages = loader.load_and_split()
        print("Completed loading PDF")
        return pages

    def read_pdf(uploaded_file):
        if uploaded_file is not None:
            os.makedirs(upload_dir, exist_ok=True)
            time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file = f"uploaded_file_{time}.pdf"
            file_path = os.path.join(upload_dir, file)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            return file_path

    def save_uploaded_file(uploaded_file, folder_path):
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path

    def load_docs(directory):
        loader = DirectoryLoader(directory,glob="**/*.pdf")
        documents = loader.load()
        return documents

    def split_docs(documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)
        return docs

    if chosen_id == "tab_pdf":
        os.makedirs(upload_dir, exist_ok=True)
        uploaded_files = st.file_uploader("Upload multiple documents", accept_multiple_files=True,type=["pdf"])
        if st.button("Upload"):
            if upload_dir:
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file, upload_dir)
                documents = load_docs(upload_dir)

                documents_with_content = [doc for doc in documents if doc.page_content is not None]
                try:
                    if not documents_with_content:
                        st.error("No documents with content loaded!")
                    else:
                        pages_doc = split_docs(documents)

                        vectordb = Chroma.from_documents(pages_doc, embeddings, persist_directory="./chromadb")
                        vectordb.persist()
                        print(pages_doc)
                        print("Successfully persisted into chroma db!")
                        st.success(f"Document(s) uploaded!")
                except:
                    st.error(f"Document(s) uploaded!")

    def perform_sentiment_analysis(comment):
        prompt = f"Open AI: Sentiment Data: {comment}\nSentiment Category: Identify the sentiment of " \
                 f"data: (Positive, Negative, Neutral)\nImprovements: (Key areas of improvement for " \
                 f"negative or neutral sentiment)\nDelight: (Key areas of delight " \
                 f"for positive sentiment)"

        response = openai.Completion.create(
            engine=DEPLOYMENT_ID,
            prompt=prompt,
            max_tokens=500,
            n=1,
            temperature=0.0,
            top_p=1.0
        )

        sentiment_data = response.choices[0].text
        sentiment_data = sentiment_data.lstrip()
        sentiment_result = sentiment_data.split('\n')

        sentiment_category = None
        improvements = None
        delight = None

        for line in sentiment_result:
            if line.startswith("Improvements:"):
                improvements = line.split(":")[1].strip()
                improvements = ', '.join(improvements.split(',')[:2]).strip()
            elif line.startswith("Delight:"):
                delight = line.split(":")[1].strip()
                delight = ', '.join(delight.split(',')[:2]).strip()

        if "Negative" in sentiment_data:
            sentiment_category = "Negative"
        elif "Positive" in sentiment_data:
            sentiment_category = "Positive"
        else:
            sentiment_category = "Neutral"

        # blob = TextBlob(comment)
        # sentiment_score = blob.sentiment.polarity
        # sentiment_score = (sentiment_score + 1) * 5

        # if sentiment_score > 5:
        #     sentiment_category = "Positive"
        # elif sentiment_score < 5:
        #     sentiment_category = "Negative"
        # else:
        #     sentiment_category = "Neutral"

        return sentiment_category, improvements, delight


    uploaded_file_csv = ""
    if chosen_id == "tab_sentiment":

        file = st.file_uploader("Upload a CSV file with Team Name, Feedback Id, and Comments:", type="csv")

        if file is not None:
            df = pd.read_csv(file)
            df['Sentiment Category'] = ''
            df['Sentiment Score'] = ''
            df['Areas of Improvement'] = ''
            df['Areas of Delight'] = ''

            df1 = df[['Feedback Id', 'Team Name', 'Comments']]

            st.write('Uploaded CSV Data')
            gd = GridOptionsBuilder.from_dataframe(df1)
            gd.configure_grid_options(domLayout='normal')
            gd.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=8)
            gd.configure_default_column(
                cellStyle={'background-color': '#3a3a4a', 'color': '#FFFFFF', 'font-size': '12px',
                           'font-family': 'EYInterstate', 'font-weight': 400},
                suppressMenu=True, wrapHeaderText=True, wrapText=True, autoHeaderHeight=True, editable=True,
                groupable=True)
            gd.configure_side_bar()
            gridop = gd.build()
            custom_css = {
                ".ag-row-hover": {"background-color": "#c4c4cd !important", "color": "#1a1a2a"},
                ".ag-header-cell-label": {"background-color": "#1a1a24 !important", "color": "#FFE600",
                                          'justify-content': 'center', "font-size": "13px",
                                          'font-family': 'EYInterstate', 'font-weight': 400},
                ".ag-row .ag-cell": {
                    'display': 'flex ',
                    'justify-content': 'center',
                    "align-items": "center"}
            }
            AgGrid(df1, gridOptions=gridop, theme='balham', fit_columns_on_grid_load=True, custom_css=custom_css)

            if st.button('Categorize'):
                for index, row in df.iterrows():
                    feedback_id = row['Feedback Id']
                    team_name = row['Team Name']
                    comment = row['Comments']
                    sentiment_category, improvements, delight = perform_sentiment_analysis(comment)

                    df.at[index, 'Sentiment Category'] = sentiment_category
                    # df.at[index, 'Sentiment Score'] = sentiment_score
                    if sentiment_category == 'Negative' or sentiment_category == 'Neutral':
                        df.at[index, 'Areas of Improvement'] = improvements if improvements else ''
                    elif sentiment_category == 'Positive':
                        df.at[index, 'Areas of Delight'] = delight if delight else ''

                output_df = df.copy()
                output_df = output_df[['Feedback Id', 'Team Name', 'Comments', 'Sentiment Category', 'Sentiment Score',
                                       'Areas of Improvement', 'Areas of Delight']]

                st.write('Categorized Feedback Data')
                gd = GridOptionsBuilder.from_dataframe(output_df)
                gd.configure_grid_options(domLayout='normal')
                gd.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=8)
                gd.configure_default_column(
                    cellStyle={'background-color': '#3a3a4a', 'color': '#FFFFFF', 'font-size': '12px',
                               'font-family': 'EYInterstate', 'font-weight': 400},
                    suppressMenu=True, wrapHeaderText=True, autoHeaderHeight=True, editable=True,
                    groupable=True)
                gd.configure_side_bar()
                gridop = gd.build()
                custom_css = {
                    ".ag-row-hover": {"background-color": "#c4c4cd !important", "color": "#1a1a2a"},
                    ".ag-header-cell-label": {"background-color": "#1a1a24 !important", "color": "#FFE600",
                                              'justify-content': 'center', "font-size": "13px",
                                              'font-family': 'EYInterstate', 'font-weight': 400},
                    ".ag-row .ag-cell": {
                        'display': 'flex ',
                        'justify-content': 'center',
                        "align-items": "center"}
                }
                AgGrid(output_df, gridOptions=gridop, theme='balham', fit_columns_on_grid_load=True,
                       custom_css=custom_css)

                st.download_button("Download Output", output_df.to_csv(index=False), file_name='classified_data.csv',
                                   mime='text/csv')

    if chosen_id == "tab_dilo":
        uploaded_file_csv = st.file_uploader("Upload DILO CSV file", type=['csv'])
        print("inside dilo tab generate response")
        # Connect to SQLite database
        conn = sqlite3.connect('employee.db')
        c = conn.cursor()
        # Create table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS dilo_desc_class
                     (ID INTEGER, description TEXT, category TEXT, description_english TEXT)''')

        if uploaded_file_csv is not None:
            print("in csv uploaded got")
            df = pd.read_csv(uploaded_file_csv)
            df = df[['ID', 'Description']]
            print(df)

            # Classify descriptions
            df['Category'] = df['Description'].apply(classify_description)

            # Display the classified descriptions
            st.write("Classified Descriptions:")
            # st.dataframe(df)
            gd = GridOptionsBuilder.from_dataframe(df)
            gd.configure_pagination(enabled=True)
            gd.configure_default_column(cellStyle={'color': '#FFFFFF', 'font-size': '12px', 'font-weight': 400},
                                        suppressMenu=True, wrapHeaderText=True, autoHeaderHeight=True, editable=True,
                                        groupable=True)
            gridop = gd.build()
            custom_css = {
                ".ag-row-hover": {"background-color": "#747480 !important"},
                ".ag-header-cell-label": {"background-color": "#1a1a24 !important", "color": "#FFE600",
                                          'justify-content': 'center', "font-size": "13px", 'font-weight': 400},
                ".ag-row .ag-cell": {
                    'display': 'flex ',
                    'justify-content': 'center',
                    "align-items": "center"}
            }
            AgGrid(df, gridOptions=gridop, height=350, theme='balham', custom_css=custom_css)

            # Insert data into SQLite database
            insert_data_into_db(df)
            st.success("You can now download the data in the link below!")
            st.download_button("Download CSV", df.to_csv(), file_name='classified_data.csv', mime='text/csv')
        else:
            placeholder = st.empty()
    else:
        placeholder = st.empty()

    # Initialize counter
    counter = 0


    def def_sqlite(prompt):
        db = SQLDatabase.from_uri("sqlite:///C:/Users/JH864DU/OneDrive - EY/Documents/My Projects/Open AI/employee.db")
        llm = OpenAI(openai_api_key=api_key, temperature=0, engine='text-davinci-003')
        db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
        resp = db_chain.run(prompt)
        return resp

    if 'generated_pdf' not in st.session_state:
        st.session_state['generated_pdf'] = ["I'm your Team's Assistant, How may I help you?"]
    if 'past_pdf' not in st.session_state:
        st.session_state['past_pdf'] = ['Hi!']

    if 'generated_util' not in st.session_state:
        st.session_state['generated_util'] = ["I'm your Team's Assistant, How may I help you?"]
    if 'past_util' not in st.session_state:
        st.session_state['past_util'] = ['Hi!']

    if chosen_id in ["tab_pdf", "tab_util"]:
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
    if chosen_id in ["tab_pdf", "tab_util"]:
        with input_container:
            user_input = get_text()


    def classify_description(description):
        prompt = f"Classify the following description into the appropriate category:\n\nDescription: {description}\nCategories = {categories}\n\nCategory:"

        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=1,
            n=1,
            temperature=0.0,
            stop=None
        )

        category_number = response.choices[0].text.strip()
        category = categories.get(category_number, 'Unknown')
        return category


    # Response output
    ## Function for taking user prompt as input followed by producing AI generated responses
    def generate_response(prompt, selected_tab):
        print(selected_tab)
        reply = ""
        if selected_tab == "tab_util":
            if prompt:
                # Handle the SQLite query and get the reply
                reply = def_sqlite(prompt)
            else:
                st.warning("Please enter a query.")
        if selected_tab == "tab_pdf":
            if prompt:
                vectordb_new = Chroma(persist_directory="./chromadb", embedding_function=embeddings)
                print("vectordb_new = ", vectordb_new)
                print("persist_directory = ", persist_directory)
                print("prompt = ", prompt)

                llm_pdf = ChatOpenAI(engine=DEPLOYMENT_ID)
                chain = load_qa_chain(llm_pdf, chain_type="stuff")
                matching_docs = vectordb_new.similarity_search(prompt)
                print("matching_docs = ", matching_docs[0].page_content)

                reply = chain.run(input_documents=matching_docs, question=prompt)
                print("reply = ", reply)
            else:
                st.warning("Please enter a query.")
        return reply


    ## Conditional display of AI generated responses as a function of user provided prompts
    if chosen_id in ["tab_pdf", "tab_util"]:
        with response_container:
            if chosen_id == "tab_pdf":
                if user_input:
                    try:
                        response = generate_response(user_input, chosen_id)
                        st.session_state.past_pdf.append(user_input)
                        st.session_state.generated_pdf.append(response)
                    except:
                        st.error(
                            "An error occurred while retrieving the generated responses. "
                            "Please try again with a different prompt.")
                if st.session_state['generated_pdf']:
                    for i in range(len(st.session_state['generated_pdf'])):
                        message(st.session_state['past_pdf'][i], is_user=True, key=str(i) + '_user')
                        message(st.session_state["generated_pdf"][i], key=str(i))

            if chosen_id == "tab_util":
                if 'past_util' not in st.session_state:
                    st.session_state['past_util'] = []
                if user_input:
                    try:
                        response = generate_response(user_input, chosen_id)
                        st.session_state.past_util.append(user_input)
                        st.session_state.generated_util.append(response)
                    except:
                        st.error(
                            "An error occurred while retrieving the generated responses. "
                            "Please try again with a different prompt.")
                if st.session_state['generated_util']:
                    num_responses = len(st.session_state['generated_util'])
                    for i in range(num_responses):
                        if i < len(st.session_state['past_util']):
                            message(st.session_state['past_util'][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated_util"][i], key=str(i))

hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

##footer

footer = """<style>
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
st.markdown(footer, unsafe_allow_html=True)

