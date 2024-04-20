from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import datetime
from langchain.llms import Anthropic
from langchain_anthropic import ChatAnthropic
import pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def authenticate_user():
    user_id = st.sidebar.text_input("Enter your User ID:")
    password = st.sidebar.text_input("Enter your password:", type="password")
    
    if user_id == "supriya" and password == "hello":
        return user_id
    else:
        return None
    
def load_transcripts(directory):
    transcripts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                transcript = file.read()
                transcripts.append(transcript)
    return transcripts

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt):
    text = txt.read().decode("utf-8")
    return text

def extract_text_from_journal(user_id):
    journal_file = f"journal_entries/{user_id}_health_journal.txt"
    entries = []
    if os.path.exists(journal_file):
        with open(journal_file, 'r', encoding='utf-8') as file:
            entry = ""
            for line in file:
                if line.startswith("Date: "):
                    if entry:
                        entries.append(entry)
                    entry = line
                else:
                    entry += line
            if entry:
                entries.append(entry)
    return entries

def parse_date(date_string):
    try:
        return datetime.datetime.strptime(date_string, "%m-%d-%Y %A")
    except ValueError:
        return datetime.datetime.strptime(date_string, "%m-%d-%Y")

def get_journal_entries(user_id):
    journal_file = f"journal_entries/{user_id}_health_journal.txt"
    entries = []
    if not os.path.exists(journal_file):
        return entries

    with open(journal_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Date: "):
                entry_date = parse_date(line[6:].strip())
                entries.append(entry_date)
    entries.sort(reverse=True)
    return entries

def read_entry(user_id, date):
    journal_file = f"journal_entries/{user_id}_health_journal.txt"
    content = ""
    with open(journal_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        if line.startswith("Date: ") and start_reading:
            break

        if start_reading:
            content += line

        if line.startswith("Date: ") and date == parse_date(line[6:].strip()):
            start_reading = True

    return content

def write_entry(user_id, date, content):
    journal_file = f"journal_entries/{user_id}_health_journal.txt"
    new_entry = f"\nDate: {date}\n{content}\n\n"

    if not os.path.exists(journal_file):
        with open(journal_file, "w", encoding="utf-8") as f:
            f.write(new_entry)
    else:
        with open(journal_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines_to_remove = set()
        removing_entry = False
        for i, line in enumerate(lines):
            if line.startswith("Date: "):
                if date == line[6:].strip():
                    removing_entry = True
                    lines_to_remove.add(i)
                else:
                    removing_entry = False

            if removing_entry:
                lines_to_remove.add(i)

        lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]

        new_entry_date = parse_date(date)
        position = None
        for i, line in enumerate(lines):
            if line.startswith("Date: "):
                entry_date = parse_date(line[6:].strip())
                if new_entry_date < entry_date:
                    position = i
                    break

        if position is None:
            lines.append(new_entry)
        else:
            lines.insert(position, new_entry)

        with open(journal_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    #os.environ["OPENAI_API_KEY"] = openai_api_key
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("HealthStory: Your personalized Health Coach")
    st.write("Record your progress, get personalized insights.")

    user_id = authenticate_user()

    if user_id is None:
        st.error("Invalid User ID or password. Please try again.")
    else:
        tabs = st.tabs(["Journal Entry", "AI Assistant"])

        with tabs[0]:
            selected_date = st.date_input("Select the date for the journal entry:", value=datetime.date.today())
            formatted_date = selected_date.strftime("%m-%d-%Y %A")
            st.write(f"Selected date: {formatted_date}")

            entry = ""

            if selected_date in get_journal_entries(user_id):
                entry = read_entry(user_id, selected_date)

            new_entry = st.text_area("Write your health journal entry:", entry)

            if st.button("Submit"):
                write_entry(user_id, formatted_date, new_entry)
                st.success("Journal entry saved successfully!")

            st.header("Previous Journal Entries")
            entries = get_journal_entries(user_id)

            if entries:
                selected_entry_date = st.selectbox("Select an entry to view:", entries, format_func=lambda x: x.strftime("%m-%d-%Y %A"))

                if st.button("Load Entry"):
                    entry_text = read_entry(user_id, selected_entry_date)
                    st.write(f"**{selected_entry_date.strftime('%m-%d-%Y %A')}**")
                    st.markdown(entry_text.replace("\n", "<br>"), unsafe_allow_html=True)

            else:
                st.write("No previous entries found.")

        with tabs[1]:
            
            
            
            #api_key = st.text_input("Enter your OpenAI API key:", type="password")
            #os.environ["OPENAI_API_KEY"] = api_key

            #if not api_key:
                #st.warning("Please enter your OpenAI API key to continue.")
            #else:
                
                file_type = st.selectbox("Choose the file type", options=["Health Journal", "Medical Report"])
                file = None
                text = None

                if file_type == "PDF":
                    file = st.file_uploader("Upload your PDF", type="pdf")
                    if file is not None:
                        text = extract_text_from_pdf(file)
                elif file_type == "TXT":
                    file = st.file_uploader("Upload your TXT", type="txt")
                    if file is not None:
                        text = extract_text_from_txt(file)
                elif file_type == "Health Journal":
                    text = extract_text_from_journal(user_id)

                if file is not None or file_type == "Health Journal":
                    if file_type == "Health Journal":
                        journal_entries = extract_text_from_journal(user_id)
                        text = "\n".join(journal_entries)  # Concatenate journal entries into a single string
                    
                    text_splitter = CharacterTextSplitter(
                        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    #pinecone.init(api_key="a7959969-443f-4b9e-a9db-40f47dce8eee", environment="us-east-1")
                    
                    pc = Pinecone(api_key="a7959969-443f-4b9e-a9db-40f47dce8eee")

                    

                    # Load podcast transcripts
                    transcript_directory = "transcripts"
                    transcripts = load_transcripts(transcript_directory)
                    index_name = "healthjournal"
                    dimension = 1536  # Dimension of the OpenAI embeddings
                    if index_name in pc.list_indexes():
                        # Use the existing index
                        index = pc.get_index(index_name)
                    else:
                        # Create a new index
                        index = pc.create_index(
                            name=index_name,
                            dimension=dimension,
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )

                        
                    

                    embeddings = OpenAIEmbeddings()
                    knowledge_base = PineconeVectorStore.from_texts(
                    chunks + transcripts,
                    embeddings,
                    index_name=index_name
                )

                    st.sidebar.header("Chatbot")
                    llm = OpenAI()
                    #llm = ChatAnthropic(temperature=0,anthropic_api_key="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", model_name="claude-3-opus-20240229")
                    window_size = 5
                    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=window_size)
                    chain = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever(), memory=memory)

                    if f"{user_id}_chat_history" not in st.session_state:
                        st.session_state[f"{user_id}_chat_history"] = []

                    def extract_keywords(question):
                        keywords = ["breakfast", "lunch", "dinner", "snack", "healthy", "nutritious"]
                        extracted_keywords = [keyword for keyword in keywords if keyword in question.lower()]
                        return extracted_keywords

                    def recommend_food_spots(keywords):
                        food_spots = {
                            "breakfast": ["Healthy Breakfast Cafe", "Oatmeal Paradise"],
                            "lunch": ["Salad Bar", "Wrap Haven"],
                            "dinner": ["Grilled Delights", "Quinoa Kitchen"],
                            "snack": ["Fruit Fantasia", "Nutty Nibbles"],
                            "healthy": ["Green Goodness", "Wholesome Eats"],
                            "nutritious": ["Nutrient Nook", "Vitamin Vault"]
                        }

                        recommended_spots = []
                        for keyword in keywords:
                            if keyword in food_spots:
                                recommended_spots.extend(food_spots[keyword])

                        return recommended_spots[:3]

                    def generate_response(user_question):
                        docs = knowledge_base.similarity_search(user_question)
                        with get_openai_callback() as cb:
                            if "food spots" in user_question.lower():
                                prompt = f"Hey there! Thanks for reaching out. Based on your question: '{user_question}', provide a relevant response and suggest healthy food spots in San Francisco."
                            elif "hiking spots" in user_question.lower():
                                prompt = f"Hey there! Thanks for reaching out. Based on your question: '{user_question}', provide a relevant response and suggest hiking trails in San Francisco."
                            else:
                                prompt = f"Hey there! Thanks for reaching out. Based on your question: '{user_question}', provide a relevant response."
                            
                            chat_history = st.session_state[f"{user_id}_chat_history"]
                            
                            response = chain({"question": prompt})["answer"]
                            #response = chain.run(input_documents=docs, question=prompt)
                            print(cb)
                            return response

                    user_question = st.sidebar.text_input("Ask a question about your health journal:")
                    if user_question:
                        response = generate_response(user_question)
                        st.session_state[f"{user_id}_chat_history"].append({"message": user_question, "is_user": True})
                        st.session_state[f"{user_id}_chat_history"].append({"message": response, "is_user": False})

                    for chat in st.session_state[f"{user_id}_chat_history"]:
                        if chat['is_user']:
                            st.sidebar.markdown(f"**You:** {chat['message']}")
                        else:
                            st.sidebar.markdown(f"**HealthStory:** {chat['message']}")

if __name__ == '__main__':
    main()