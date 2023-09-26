import os
from langchain.vectorstores import ElasticsearchStore
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import os

ES_URL = os.getenv('ES_URL')
ES_INDEX = os.getenv('ES_INDEX')
ES_URL = os.getenv('ES_URL')
ES_API_KEY = os.getenv('ES_API_KEY')

store = ElasticsearchStore(
    index_name=ES_INDEX,
    es_url=ES_URL,
    es_api_key=ES_API_KEY,
    query_field="text_field",
    vector_query_field="vector_query_field.predicted_value",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(
        query_model_id="sentence-transformers__all-minilm-l6-v2"
    )
)


def load_llm():
    model_id = './models/google_flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return HuggingFacePipeline(pipeline=pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    ))


def get_chain():
    template_informed = """
   I am a helpful AI that answers questions.
   When I don't know the answer I say I don't know.
   I know context: {context}
   when asked: {question}
   my response using only information in the context is: """
    prompt_informed = PromptTemplate(
        template=template_informed,
        input_variables=["context", "question"])

    return LLMChain(prompt=prompt_informed, llm=load_llm())


# Getting the Model Chain
model_chain = get_chain()

# Question & Answer using LLM
print('I am a Q&A droid, ask me any question about Star Wars!')
question = input("User Question: ")

# Perform Elasticsearch
document = store.similarity_search(question, k=1).pop().page_content

print(model_chain.run(
    context=document,
    question=question
))
