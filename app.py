import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys
import fitz
from dotenv import load_dotenv, find_dotenv

# gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from logger import logging
from exception import customexception

# loading environment variables
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """ 
    Extracts the text content from a list of PDF documents.

    This function takes a list of PDF file paths, reads each PDF file, 
    and extracts the text from all the pages in each document. The 
    extracted text is concatenated into a single string and returned.

    Args:
        pdf_docs (list): A list of file paths (strings) to the PDF documents 
                          from which text needs to be extracted.

    Returns:
        str: A string containing the concatenated text extracted from all the 
             pages of the provided PDF documents.

    Raises:
        customexception: If any exception occurs during the text extraction 
                         process, the exception is logged and raised as 
                         `customexception`.

    Example:
        pdf_files = ['document1.pdf', 'document2.pdf']
        text = get_pdf_text(pdf_files)
        print(text)

    Notes:
        - This function uses the `PdfReader` from the `PyPDF2` library to 
          extract text from each PDF page.
        - The function logs the start and completion of the text extraction 
          process, and it also logs any exceptions that occur during the 
          process.
    """
    try:

        logging.info(f"[-] Text extraction started...")

        text = ""
        # for pdf in pdf_docs:
        #     pdf_reader=PdfReader(pdf)
        #     for page in pdf_reader.pages:
        #         text += page.extract_text()

        # docs = fitz.open(pdf_docs)
        docs = fitz.open(stream=pdf_docs.read(), filetype="pdf")

        for page_num in range(len(docs)):
            page = docs[page_num]
            text += page.get_text("text") + "\n"

        logging.info(f"[*] Text extrcation completed!")

        return text
    
    except Exception as e:

        logging.info(f"[!] Exception while getting text from PDF : {e}")

        raise customexception(e, sys)

def get_text_chunks(text):
    """
    Splits a large text into smaller, manageable chunks.

    This function takes a long text string and splits it into chunks of 
    a specified size using a recursive character-based text splitter. 
    The chunks are designed to have a maximum size of `chunk_size` 
    with a possible overlap (`chunk_overlap`) between consecutive chunks.

    Args:
        text (str): The large text string that needs to be split into chunks.

    Returns:
        list: A list of text chunks (strings), where each chunk is a substring
              of the original text. The chunks have a maximum size of `chunk_size`, 
              with a `chunk_overlap` between adjacent chunks.

    Raises:
        customexception: If any error occurs during the text chunking process, 
                         an exception is logged and raised as `customexception`.
    
    Example:
        large_text = "A very long text that needs to be chunked ..."
        text_chunks = get_text_chunks(large_text)
        print(text_chunks)

    Notes:
        - This function uses `RecursiveCharacterTextSplitter` to split the input text.
        - `chunk_size` is set to 10,000 characters, and `chunk_overlap` is set to 1,000 
          characters by default.
        - The function logs the start and completion of the chunking process, 
          and it also logs any exceptions that occur.   
    """
    try:
        logging.info(f"[-] Text chunking started...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)

        logging.info("[*] Text chunking completed!")

        return chunks
    
    except Exception as e:

        logging.info(f"[!] Exception while converting text to chunks : {e}")

        raise customexception(e, sys)

def get_vector_store(text_chunks):
    """ 
    Converts text chunks into a vector store using embeddings and saves it locally.

    This function takes a list of text chunks, converts each chunk into a vector 
    using a pre-trained embedding model (Google Generative AI Embeddings), and 
    stores the resulting vectors in a FAISS index. The index is then saved to the 
    local file system for future use.

    Args:
        text_chunks (list): A list of text strings (chunks) to be converted into 
                             vector embeddings.

    Returns:
        None: This function does not return any value. Instead, it saves the 
              resulting FAISS index to disk.

    Raises:
        customexception: If any error occurs during the embedding or vector store 
                         creation process, an exception is logged and raised as 
                         `customexception`.

    Example:
        text_chunks = ["This is the first chunk of text.", "This is the second chunk."]
        get_vector_store(text_chunks)  # This saves the FAISS index locally as "faiss_index"

    Notes:
        - This function uses `GoogleGenerativeAIEmbeddings` to generate embeddings for each text chunk.
        - It uses FAISS to create a vector store and saves the index to a file named "faiss_index".
        - The function logs the start and completion of the embedding conversion process, 
          and also logs any exceptions that occur during the process.
    """
    try:
        logging.info("[-] Conversion to embedding started...")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

        logging.info("[*] Conversion to embedding completed!")

    except Exception as e:

        logging.info(f"[!] Exception while converting to vector : {e}")

        raise customexception(e, sys)

def get_conversational_chain():
    """ 
    Creates and returns a conversational chain for question-answering using a large language model (LLM).

    This function initializes a conversational chain designed to answer questions based on a provided context. 
    It sets up a prompt template for the question-answering model, uses a pre-trained Google Generative AI model 
    (specified by the "gemini-pro" model), and configures the chain to process context and questions. 
    The function returns the initialized chain that can be used for answering questions with the provided context.

    Returns:
        Chain: A conversational chain (`load_qa_chain`) configured with a prompt template and a model, ready to 
               answer questions based on the provided context.

    Raises:
        customexception: If any error occurs while creating or setting up the conversational chain, an exception is 
                         logged and raised as `customexception`.

    Example:
        # Example of using the conversational chain
        chain = get_conversational_chain()
        result = chain.run(context="Your context here", question="Your question here")
        print(result)

    Notes:
        - The function uses the `ChatGoogleGenerativeAI` model with `gemini-pro` as the LLM for answering the questions.
        - The prompt template is designed to instruct the model to provide detailed answers based on the context provided.
        - The function logs the start and completion of setting up the conversation chain, and it also logs any exceptions 
          that occur during the setup.
    """
    try:
        logging.info("[-] Conversation chain started...")

        prompt_template= """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answers\n\n
        Context:\n {context}?\n
        Question:\n {question}\n

        Answer: 

        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

        prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        logging.info("[*] Conversation chain completed!")

        return chain
    
    except Exception as e:

        logging.info(f"[!] Exception while conversing with llm : {e}")
        
        raise customexception(e, sys)

def user_input(user_question):
    """ 
    Processes user input by performing a similarity search on a pre-trained FAISS index 
    and generates a response using a conversational chain.

    This function takes a user’s question, performs a similarity search on a locally saved 
    FAISS index to retrieve relevant context, and then uses a conversational chain powered 
    by a language model to generate a detailed answer. The response is then printed and 
    displayed using Streamlit (`st.write`).

    Args:
        user_question (str): The question input by the user that will be used to 
                              perform a similarity search and generate a response.

    Returns:
        None: The function does not return a value, but it prints the response to 
              the console and displays it on the web interface using Streamlit.

    Raises:
        customexception: If any error occurs during loading the FAISS index, performing 
                         the similarity search, or generating the response, an exception 
                         is logged and raised as `customexception`.

    Example:
        user_question = "What is the capital of France?"
        user_input(user_question)

    Notes:
        - The function loads the FAISS index stored locally as "faiss_index" and uses the 
          `GoogleGenerativeAIEmbeddings` model to perform the similarity search.
        - It then passes the relevant documents retrieved from the search along with the user’s 
          question to a conversational chain for response generation.
        - The final response is printed and displayed using Streamlit's `st.write` function.
    """
    try:
        logging.info("[-] Processing user input...")

        # loading googles embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # loading faiss index from the local
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # then performing similarity search
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs,
            "question": user_question},
            return_only_outputs=True
        )

        print(response)

        st.write("Reply: ", response["output_text"])

        logging.info("[*] Processing use input completed!")

    except Exception as e:
        logging.info(f"[!] Exception while processing user input : {e}")
        raise customexception(e, sys)


def main():
    """ 
    Streamlit app for chatting with PDF documents using a language model.

    This function sets up a Streamlit web application where users can upload PDF files, 
    ask questions related to the contents of the PDFs, and receive answers generated by 
    a language model. The app allows the user to upload PDFs, processes the text, converts 
    it into embeddings, and uses a conversational chain to respond to user queries.

    Functionality:
        - Sets up the main page with a header and a text input for user questions.
        - Allows users to upload PDF files from the sidebar.
        - Once a PDF is uploaded, the text is extracted, chunked, and embedded into a vector store.
        - Users can submit questions, which are answered based on the processed PDF content.

    Workflow:
        1. User uploads PDF files and submits them for processing.
        2. Text is extracted from the PDFs, split into chunks, and stored in a vector store.
        3. User asks a question, and the system retrieves relevant context from the PDF.
        4. The question is answered using a conversational chain powered by a language model.
    
    Notes:
        - The app uses Streamlit’s UI components like `file_uploader`, `text_input`, and `button` for interaction.
        - The backend processes include extracting text from PDFs, chunking it, creating embeddings, and storing them in FAISS.
    """
    try:
        # streamlit app
        st.set_page_config("Chat PDF")
        # header for front end
        st.header("Chat with PDF Using Gemini")
        # user will provide a question
        user_question = st.text_input("Ask a Question from the PDF Files")

        # function will execute qith user question
        if user_question:
            user_input(user_question)

        # side bar to upload pdf
        with st.sidebar:

            # title of side bar
            st.title("Menu:")

            # pdf uploader button
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submission Button")

            print(f"Loaded file: {pdf_docs}")

            # submit button to submit pdf
            if st.button("Submit & Process"):

                # text on sppiner
                with st.spinner("Processing..."):

                    # following flow will execute once pdf is uploaded / will log this each step for debugging purpose
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)

                    # once all process complete, message will be showen
                    st.success("Done")
    except Exception as e:
        logging.info(f"[!] Exception while loading frontend : {e}")
        raise customexception(e, sys)

if __name__ == "__main__":
    main()