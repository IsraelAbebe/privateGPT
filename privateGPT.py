#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
from omegaconf import OmegaConf


from constants import CHROMA_SETTINGS

def main(embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks):
    # Parse the command line arguments
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if not conf.get('mute_stream') else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= conf.get('hide_source')==None)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if not conf.get('hide_source') else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)



if __name__ == "__main__":
    if not load_dotenv():
        print("Could not load .env file or it is empty. Please check if it exists and is readable.")
        exit(1)

    conf = OmegaConf.from_cli()
    print(conf)
    
    embeddings_model_name = conf.get('EMBEDDINGS_MODEL_NAME',os.environ.get("EMBEDDINGS_MODEL_NAME")) 
    persist_directory = conf.get('PERSIST_DIRECTORY', os.environ.get('PERSIST_DIRECTORY'))
    
    model_type = conf.get('MODEL_TYPE', os.environ.get('MODEL_TYPE')) 
    model_path = conf.get('MODEL_PATH', os.environ.get('MODEL_PATH'))
    model_n_ctx = conf.get('MODEL_N_CTX', os.environ.get('MODEL_N_CTX')) 
    model_n_batch = int(conf.get('MODEL_N_BATCH',os.environ.get('MODEL_N_BATCH',8)))
    target_source_chunks = int(conf.get('TARGET_SOURCE_CHUNKS', os.environ.get('TARGET_SOURCE_CHUNKS',4))) 

    # python privateGPT.py PERSIST_DIRECTORY=db1
    print(embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks)
    main(embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks)
