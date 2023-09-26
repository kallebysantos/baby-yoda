import os
import pickle
from langchain.vectorstores import ElasticsearchStore
from pathlib import Path
from tqdm import tqdm

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


# Setup a Ingest Pipeline to perform the embedding
# of the text field
def create_pipeline():
    store.client.ingest.put_pipeline(
        id="input_embedding_pipeline",
        processors=[
            {
                "inference": {
                    "model_id": "sentence-transformers__all-minilm-l6-v2",
                    "field_map": {"query_field": "text_field"},
                    "target_field": "vector_query_field",
                }
            }
        ],
    )


# Creating a new index with the pipeline,
# not relying on langchain to create the index
def create_index():
    store.client.indices.create(
        index=ES_INDEX,
        mappings={
            "properties": {
                "text_field": {"type": "text"},
                "vector_query_field": {
                    "properties": {
                        "predicted_value": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "l2_norm",
                        }
                    }
                },
            }
        },
        settings={"index": {"default_pipeline": "input_embedding_pipeline"}},
    )


def load_dataset():
    batch_text = []
    file_name_pattern = "starwars_*_data*.pickle"
    files = sorted(Path('./dataset').glob(file_name_pattern))
    for fn in files:
        with open(fn, 'rb') as f:
            part = pickle.load(f)
            for _, (_, value) in tqdm(enumerate(part.items()), total=len(part)):
                paragraphs = value['paragraph']
                for p in paragraphs:
                    batch_text.append(p)

            print(f"Items found: {len(batch_text)} in {fn.name}")
            print("Trying to upload to Elasticsearch\nIf some 'Time out' error occurs just wait from Kibana dashboard until all items be ingested.")

            store.from_texts(
                texts=batch_text,
                index_name=ES_INDEX,
                es_url=ES_URL,
                es_api_key=ES_API_KEY,
                query_field="text_field",
                vector_query_field="vector_query_field.predicted_value",
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    query_model_id="sentence-transformers__all-minilm-l6-v2"
                ))


create_pipeline()
print("Pipeline created!")

create_index()
print("Index created!")

load_dataset()
print("Texts uploaded!")
