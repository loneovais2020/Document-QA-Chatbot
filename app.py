# --------------------------------------------------------------------------------------------------------
# CURRENT ISSUES:- 
# integrate memory
# while processing images shows pytesseract error 
# ----------------------------------------------------------------------------------------------------------
# ADDITIONAL FEATURES:-
# TRANSLATION 
# TEXT TO SPEECH AND SPEECH TO TEXT

import os      
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredURLLoader

from langchain.embeddings import HuggingFaceEmbeddings


from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import re


template = """
You are an AI assistant that should only use the provided context to answer questions. Do not use any other knowledge you may have. Only information from the given context should be used. It is much more important to stay within the provided context than to answer the question if the context does not contain the information needed. Do not speculate or guess an answer if the context does not provide the information required.

For example, if the context is about animals and the question is 'What color is the sky?' answering 'The sky is blue' would be completely wrong and out of context since the context is about animals, not the sky.

Use the following context to answer the question below. Reference the context in your response to justify how your answer relies only on the information provided. If the context does not contain enough information to answer the question properly, simply say 'I do not know based on the provided context.'

if the user asks you anything that requires the access to the previous answers, simply say 'I have limited memory, so I can't anwer this question'.


Also never ever spit out the context directly you are using for answering.

Context: {context}
Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["context","question"])


os.environ['API_KEY'] = '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'


# model_id = 'mistralai/Mistral-7B-Instruct-v0.1' #gives good results
model_id = 'HuggingFaceH4/zephyr-7b-alpha' #very fast and good results

llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.2,"max_new_tokens":2000})

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm_chain = LLMChain(prompt=prompt, llm=llm)




def process_url(url):
    urls=[]
    urls.append(url)
    extracted_text=""
    loaders = UnstructuredURLLoader(urls=urls)
    my_data = loaders.load()
    for data in my_data:
        extracted_text+=data.page_content
    return extracted_text





def process_text(file):
    extracted_text=""
    doc_loader=UnstructuredFileLoader(file)
    docs=doc_loader.load()
    for doc in docs:
        extracted_text+=doc.page_content

    return extracted_text


def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)
    
    chunks=text_splitter.split_text(text)
    return chunks

    


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_vectorstore_url(text_chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vectorstore=FAISS.from_documents(text_chunks,embeddings)
    return vectorstore






# -------------------------------Langchain implementation End---------------------------------------

app=Flask(__name__)
app.secret_key = "secret key" 



path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'pptx', 'jpg','png','docx','eml','html'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def delete_uploads():
    uploads_path = os.path.join(app.root_path, UPLOAD_FOLDER)
    for filename in os.listdir(uploads_path):
        file_path = os.path.join(uploads_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_files():
    global extracted_text
    global vectorstore
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        url_input = request.form.get('input')

        # Check if either a file or a URL is provided, but not both
        if uploaded_file and url_input:
            message='Please provide either a file or a URL, not both'
            return render_template("index.html",message=message)
        
        elif not uploaded_file and not url_input:
            message='Please provide a file or a URL'
            return render_template("index.html",message=message)

        elif uploaded_file:
            # Handle file upload
            if allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                filename = filename.replace(' ', '')
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                text = process_text(f"uploads/{filename}")
                delete_uploads()
                text_chunks = get_text_chunks(text)
                


            else:
                message='Allowed file types are pdf, doc, docx, txt, ppt'
                return render_template("index.html",message=message)
        elif url_input:
            
           
            # Handle URL input
            text=process_url(url_input)
            text_chunks = get_text_chunks(text)




            
        # create vector store        
        vectorstore = get_vectorstore(text_chunks)
        

        return render_template('chat.html')  





@app.route('/get')
def get_bot_response():

  # Get user question
  user_question = request.args.get('msg')

 


  if user_question:
       
       context = vectorstore.similarity_search(user_question)
    #    answer = llm_chain.run({"context":context,"question":user_question})
    #    answer=re.sub(r'\s*Answer: \s*', '', answer)
    #    bot_msg = f"{answer}"

       try:
           answer = llm_chain.run({"context":context,"question":user_question})
           answer=re.sub(r'\s*Answer: \s*', '', answer)
           bot_msg = answer
       
       except Exception as err:
           bot_msg=f'Exception occurred. Please try again {str(err)}'
           


       return bot_msg




if __name__ == "__main__":
    app.run(debug=True)









