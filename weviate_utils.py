import weaviate
import uuid
from weaviate.auth import AuthApiKey
from embed_utils import get_embeddings
from datetime import datetime, timezone
from tqdm import tqdm
from weaviate.classes.config import Property, DataType, Configure
from weaviate.exceptions import UnexpectedStatusCodeError


WEAVIATE_URL = "xmblhicrrusiwmvufnun6g.c0.asia-southeast1.gcp.weaviate.cloud"  # replace with your cluster URL
WEAVIATE_API_KEY = "eW1MakJBa1cyY3BBWW5IT196VENQZS9LUHg4VkEvMUZ0WUpCQkRocDluSlJSNnh5YTNreGVRcEFucDVzPV92MjAw"
COLLECTION_NAME = "newlearn"


def get_client():
    """Return a connected Weaviate v4 client"""
    auth = AuthApiKey(WEAVIATE_API_KEY)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth
    )
    return client


def get_collection(client, collection_name=COLLECTION_NAME):
    """
    Returns the existing collection.
    """
    return client.collections.get(collection_name)




def upsert_docs(class_name, docs, file, project_id, tags=None):
    """
    Upsert PDF chunks into Weaviate v4.x using batch insertion.
    """

    # collection_name='newlearn'
    client = get_client()
    collection = client.collections.get(class_name)

    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", None)

        vector = get_embeddings(doc.page_content)  # manual embedding

        collection.data.insert(
            properties={
               "document_id": str(uuid.uuid4()),
                "source_uri": file.filename,
                "project_id": project_id,
                "tags": tags.split(",") if tags else [],
                "chunk_index": i,
                "page_number": doc.metadata.get("page", None),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content": doc.page_content
            },
            vector=vector
        )

    
    # try:
    #     for i, d in enumerate(docs):
    #         vector = get_embeddings(d.page_content)
    #         properties = {
    #             "document_id": str(uuid.uuid4()),
    #             "source_uri": file.filename,
    #             "project_id": project_id,
    #             "tags": tags.split(",") if tags else [],
    #             "chunk_index": i,
    #             "page_number": d.metadata.get("page", None),
    #             "created_at": datetime.utcnow().isoformat(),
    #             "content": d.page_content
    #         }

    #         # v4 method: add object with vector
    #         client.data_object.create(
    #             properties,
    #             class_name,
    #             vector=vector
    #         )
    #     batch.flush()  # flush all objects to Weaviate
        print(f"Inserted {len(docs)} chunks for file {file.filename} ✅")

    # except Exception as e:
    #     print(f"Error upserting docs: {e}")

    # finally:
    #     client.close()  # prevent ResourceWarning

def query_docs(client, collection, query_text, source_uri=None, project_id=None, tags=None, top_k=5):
    """
    Queries Weaviate with optional filters.
    """
    try:
        query_vector = get_embeddings(query_text)

        # Build filter dynamically
        filters = None
        if source_uri:
            filters = Filter.by_property("source_uri").equal(source_uri)
        elif project_id:
            filters = Filter.by_property("project_id").equal(project_id)
        elif tags:
            filters = Filter.by_property("tags").contains_any(tags)

        # Search
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=filters
        )

        return [
            {
                "content": r.properties.get("content"),
                "source_uri": r.properties.get("source_uri"),
                "page_number": r.properties.get("page_number"),
                "tags": r.properties.get("tags", [])
            }
            for r in results.objects
        ]

    except Exception as e:
        print(f"Error querying docs: {e}")
        return []
                                                                
