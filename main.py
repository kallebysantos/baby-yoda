import os
from langchain.vectorstores import ElasticsearchStore

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

# Perform search
documents = store.similarity_search("Who is the Mandalorian?", k=5)
print(documents)
