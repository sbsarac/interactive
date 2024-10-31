import logging
import os
import json
import requests
from datetime import datetime
import azure.functions as func

# Environment variables
AOAI_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_key = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY")
search_api_version = '2023-07-01-Preview'
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    messages = json.loads(req.get_body())

    user_query = messages[-1]['content']
    query_embedding = generate_embeddings(user_query)

    relevant_info = search_knowledge_base(query_embedding)

    messages.append({
        "role": "system",
        "content": f"Relevant information from the knowledge base: {relevant_info}"
    })

    response = chat_complete(messages)

    response_message = response["choices"][0]["message"]

    messages.append({'role': response_message['role'], 'content': response_message['content']})

    logging.info(json.dumps(response_message))

    response_object = {
        "messages": messages,
        "products": []
    }

    return func.HttpResponse(
        json.dumps(response_object),
        status_code=200
    )

def generate_embeddings(text):
    url = f"{AOAI_endpoint}/openai/deployments/{embeddings_deployment}/embeddings?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_key,
    }

    data = {"input": text}

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    return response['data'][0]['embedding']

def search_knowledge_base(vector, top_k=5):
    url = f"{search_endpoint}/indexes/{search_index_name}/docs/search?api-version={search_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": f"{search_key}",
    }

    data = {
        "vectors": [
            {
                "value": vector,
                "fields": "contentVector",
                "k": top_k
            }
        ],
        "select": "content,url,filepath,title,meta_json_string",
        "top": top_k
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error in search_knowledge_base: {str(e)}")
        return json.dumps({"error": f"Failed to retrieve information from the knowledge base: {str(e)}"})

    relevant_info = []
    if 'value' in results:
        for item in results['value']:
            relevant_info.append({
                "content": item.get('content'),
                "url": item.get('url'),
                "filepath": item.get('filepath'),
                "title": item.get('title'),
                "meta_json_string": item.get('meta_json_string')
            })
    else:
        logging.warning(f"Unexpected response format from search_knowledge_base: {results}")
        return json.dumps({"warning": "No relevant information found in the knowledge base"})

    return json.dumps(relevant_info)

def chat_complete(messages):
    url = f"{AOAI_endpoint}/openai/deployments/{chat_deployment}/chat/completions?api-version={AOAI_api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AOAI_key
    }

    data = {
        "messages": messages,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()

    return response