import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from hypercube.hotpotqa.hypercube_cultural_product import retriever_hypercube_cultural_product
from hypercube.hotpotqa.hypercube_person import retriever_hypercube_person
from hypercube.hotpotqa.hypercube_location import retriever_hypercube_location


# Set up your API key and LLM
import os
os.environ["OPENAI_API_KEY"] = ""


# Be sure to have your OPENAI_API_KEY set as an environment variable
llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)  # gpt-4o-mini-2024-07-18, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07, gpt-5-2025-08-07



def route_query_to_retrieve_docs_hotpotqa(query, corpus, k):
    router_prompt = """
    Analyze the following query. 
    If the query is centered with cultural product (story, film, movie, song, magzine, book, award, opera, album, sport, match, competition, act, and game), e.g., who wrote The Adventure of the Seven Clocks? You need to return 'CULTURAL_PRODUCT' since the key information is related to the story of The Adventure of the Seven Clocks.
    If the query is centered with person, e.g., when was Nicki Minaj born or how old is the performer or director, you need to return 'PERSON', since the key information is related to the person Nicki Minaj, performer or director.
    If the query is centered with location (asking specific places or geographic information), e.g., where was Marufabad and Nasamkhrali, you need to return 'LOCATION' since the key information is related to the location Marufabad and Nasamkhrali.

    Query: {query}

    Only output "CULTURAL_PRODUCT" or "PERSON" or "LOCATION".
    """
    return_docs, return_doc_ids = '', ''

    # Use the LLM to make the decision
    cube_decision = llm.invoke(router_prompt.format(query=query)).content.strip()
    print(f"Assigned cube: {cube_decision}")

    if cube_decision == "CULTURAL_PRODUCT":
        return_docs, return_doc_ids = retriever_hypercube_cultural_product(query, corpus, k)

    elif cube_decision == "PERSON":
        return_docs, return_doc_ids = retriever_hypercube_person(query, corpus, k)

    elif cube_decision == "LOCATION":
        return_docs, return_doc_ids = retriever_hypercube_location(query, corpus, k)

    return return_docs, return_doc_ids