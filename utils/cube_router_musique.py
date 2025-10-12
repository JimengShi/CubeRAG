import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from hypercube.musique.hypercube_cultural_product import retriever_hypercube_cultural_product
from hypercube.musique.hypercube_person import retriever_hypercube_person
from hypercube.musique.hypercube_location import retriever_hypercube_location
from hypercube.musique.hypercube_politics import retriever_hypercube_politics


# Set up your API key and LLM
import os
os.environ["OPENAI_API_KEY"] = "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"


# Be sure to have your OPENAI_API_KEY set as an environment variable
llm = ChatOpenAI(model="gpt-5-2025-08-07", temperature=0)  # gpt-4o-mini-2024-07-18, gpt-5-mini-2025-08-07, gpt-5-nano-2025-08-07, gpt-5-2025-08-07



def route_query_to_retrieve_docs_musique(query, corpus, k):
    router_prompt = """
    Analyze the following query. Determine the main category or categories based on the most explicit entities or keywords in the query.
    If the query is centered with cultural product (story, film, movie, song, magzine, book, award, opera, album, sport, match, competition, act, and game), e.g., who wrote The Adventure of the Seven Clocks? who is the performer of "Till dom ensamma", You need to return 'CULTURAL_PRODUCT' since the key information The Adventure of the Seven Clocks and Till dom ensamma are explicit, and they are cultural products.
    If the query is centered with person, e.g., when was Nicki Minaj born or how old is the performer or director, you need to return 'PERSON', since the key information Nicki Minaj, performer or director are explicit, and they are person names.
    If the query is centered with location (asking specific places, river, hill, company, or geographic information), e.g., where was Marufabad and Nasamkhrali, who was in charge of North Holland? you need to return 'LOCATION' since the key information Marufabad, Nasamkhrali, North Holland are explicit, and they are locations.
    If the query is related to politics, e.g., governmental rules, agencies, you need to return 'POLITICS'.

    Query: {query}

    Only output "CULTURAL_PRODUCT" or "PERSON" or "LOCATION" or "POLITICS".
    """
    return_docs, return_doc_ids = '', ''

    # Use the LLM to make the decision
    cube_decision = llm.invoke(router_prompt.format(query=query)).content.strip()
    print(f"Assigned cube: {cube_decision}")

    if cube_decision == "CULTURAL_PRODUCT":
        return_docs, return_doc_ids = retriever_hypercube_cultural_product(query, corpus, k)

    elif cube_decision == "PERSON":
        return_docs, return_doc_ids = retriever_hypercube_person(query, corpus, k)

    # elif cube_decision == "LOCATION":
    #     return_docs, return_doc_ids = retriever_hypercube_location(query, corpus, k)

    elif cube_decision == "POLITICS":
        return_docs, return_doc_ids = retriever_hypercube_politics(query, corpus, k)

    return return_docs, return_doc_ids


# def route_query_to_retrieve_docs_musique(query, corpus, k):
#     router_prompt = """
#     Analyze the following query. Determine the main category or categories relevant to the query.

#     Categories:
#     1. CULTURAL_PRODUCT: If the query is centered on a cultural product (story, film, movie, song, magazine, book, award, opera, album, sport, match, competition, act, or game), e.g., "who wrote The Adventure of the Seven Clocks?".
#     2. PERSON: If the query is centered on a person (e.g., birth date, age, profession of an individual, performer, or director), e.g., "when was Nicki Minaj born or how old is the performer or director?".
#     3. LOCATION: If the query is centered on a location (asking specific places, river, hill, company, or geographic information), e.g., "where was Marufabad and Nasamkhrali?", "Who is the designer of the Southeast Library?".

#     Query: {query}

#     Output one or two categories, separated by a comma and no spaces (e.g., CULTURAL_PRODUCT,PERSON).
#     If only one category is relevant, output only that category (e.g., LOCATION).
#     Only output the category names.
#     """
    
#     # Initialize containers for combined results
#     combined_docs = ''
#     combined_doc_ids = []

#     # Use the LLM to make the decision
#     raw_decision = llm.invoke(router_prompt.format(query=query)).content.strip().upper()
    
#     # 1. Parse the decision into a list
#     # e.g., "CULTURAL_PRODUCT,PERSON" -> ["CULTURAL_PRODUCT", "PERSON"]
#     cube_decisions = [d.strip() for d in raw_decision.split(',') if d.strip()]
    
#     print(f"Assigned cubes: {cube_decisions}")

#     # 2. Iterate through all decisions and execute the corresponding function
#     for decision in cube_decisions:
        
#         return_docs = ''
#         return_doc_ids = ''

#         if decision == "CULTURAL_PRODUCT":
#             return_docs, return_doc_ids = retriever_hypercube_cultural_product(query, corpus, k)

#         elif decision == "PERSON":
#             return_docs, return_doc_ids = retriever_hypercube_person(query, corpus, k)

#         elif decision == "LOCATION":
#             return_docs, return_doc_ids = retriever_hypercube_location(query, corpus, k)

#         # 3. Combine the results
#         combined_docs = "\n\n".join(return_docs)
#         combined_doc_ids.extend(return_doc_ids)
    

#     return combined_docs, combined_doc_ids