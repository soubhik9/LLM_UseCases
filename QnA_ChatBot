import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load documents
loader = TextLoader('/content/spiderman.txt')
documents = loader.load()

# 2. Create LangChain-compatible embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')

# 3. Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(documents, embedding_model)

# 4. Load local language model (e.g., GPT-2)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 5. Create a text-generation pipeline with proper config
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,         # Controls output length
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# 6. Wrap the generator with LangChain's LLM interface
llm = HuggingFacePipeline(pipeline=generator)

# 7. Set up memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 8. Create the ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 9. Ask a question
query = "who is Peter Parker?"
response = qa_chain.run({"question": query, "chat_history": []})
print(response)
