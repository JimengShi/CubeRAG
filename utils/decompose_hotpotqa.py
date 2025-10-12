from openai import OpenAI
import os
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



model = SentenceTransformer('intfloat/e5-base-v2')


from pydantic import (
    BaseModel,
    Field
)

from typing import (
    List, 
    Literal
)


ent2emb = None
def embed_string(target_str, emb_model, dict_path):
    global ent2emb
    if ent2emb is None:
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                ent2emb = pickle.load(f)      
        else:
            ent2emb = {}
            print(f"No saved dictionary found, initialized a new dict from string to corresponding embedding.")

    if target_str in ent2emb:
        return 
    else:
        ent2emb[target_str] = emb_model.encode('query: ' + target_str, normalize_embeddings=True)
        with open(dict_path, 'wb') as f:
            pickle.dump(ent2emb, f)


def decompose_query_along_cultural_product(query, dimensions):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = (
        f"You are an expert on question understanding. "
        f"Your task is to:\n"
        f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
        f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
        
        f"cultural_product_name dimension can be specific names/entities/phrases/terms of story, film, movie, song, magzine, book, award, opera, album, sport, match, competition, and game. Feel free to include more cultural products if you find any.\n"
        f"cultural_product_type dimension can be types of cultural_products, such as film, movie, song, magzine, book, award, opera, album, sport, match, competition, and game. Feel free to include more cultural product types if you find any.\n"
        f"action_relation dimension can be any actions or relations that describe the entities/phrases/terms in the above cultural_product dimension.\n"

        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
        f"Example Input:\n"
        f"Question: Who is the director of film Days And Hours? or who directed the film Days And Hours? or when the film Days And Hours released?\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'cultural_product_name'; query_content: 'Days And Hours';\n"
        f"Query 2:\n"
        f"query_dimension: 'cultural_product_type'; query_content: 'film';\n"
        f"Query 3:\n"
        f"query_dimension: 'action_relation'; query_content: 'directed';\n"
        f"Query 4:\n"
        f"query_dimension: 'action_relation'; query_content: 'released';\n"
    )

    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents'
        )
        query_dimension: Literal['cultural_product_name', 'cultural_product_type', 'action_relation'] = Field(
            ...,
            title='Dimension of the entity or phrase to query documents'
        )
        
    class AllQueries(BaseModel):
        list_of_queries: List[Query] = Field(
            ...,
            title='List of queries following the required format based on the question comprehension'
        )
        
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_prompt}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )

    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa_product.pkl', )
            # embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None
    


def decompose_query_along_person(query, dimensions):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = (
        f"You are an expert on question understanding. "
        f"Your task is to:\n"
        f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
        f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
        
        f"person dimension can be person names/entities/phrases/terms.\n"
        f"role dimension can be director, composer, writer, actor, actress, musician, detective, star, player, sport player, commentator, captain. Feel free to include more roles if you find any.\n"
        f"behavior dimension can be die on, born on, birth, married to, study, work at, work as, death, located at entities/phrases/terms. Feel free to include more behaviors of person if you find any.\n"
        f"nationality dimension can be such as come from, is from, the country is. Feel free to include more nationalities if you find any.\n"

        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
        f"Example Input:\n"
        f"Question: Who is the spouse of Pjer Zalica and when did she die, and where is she from?\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'person'; query_content: 'Pjer Zalica';\n"
        f"Query 2:\n"
        f"query_dimension: 'role'; query_content: 'spouse';\n"
        f"Query 3:\n"
        f"query_dimension: 'behavior'; query_content: 'die on';\n"
        f"Query 4:\n"
        f"query_dimension: 'nationality'; query_content: 'is from';\n"
    )

    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents'
        )
        query_dimension: Literal['person', 'role', 'behavior', 'nationality'] = Field(
            ...,
            title='Dimension of the entity or phrase to query documents'
        )
        
    class AllQueries(BaseModel):
        list_of_queries: List[Query] = Field(
            ...,
            title='List of queries following the required format based on the question comprehension'
        )
        
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_prompt}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )

    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa_person.pkl', )
            # embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None
    

def decompose_query_along_location(query, dimensions):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system_prompt = (
        f"You are an expert on question understanding. "
        f"Your task is to:\n"
        f"1. **Comprehend the given question**: understand what the question asks, how to answer it step by step, and all concepts, aspects, or directions that are relevant to each step.\n"
        f"2. **Compose queries to retrieve documents for answering the question**: each document are indexed by the entities or phrases occurred inside and those entities or phrases lie within following dimensions: {dimensions}. "
        
        f"location_name dimension can be specific names/entities/phrases/terms of pecific geographical location names, highway names, church, airport, university or school. Feel free to include more geographical location names if you find any\n"
        f"location_type dimension can be street, village, city, county, state, country, highway, airport, church. Feel free to include more location types if you find any.\n"
        f"location_relation dimension can be located in, located at, belongs to, is a part of. Feel free to include more.\n"

        f"For each of the above dimension, synthesize queries that are informative, self-complete, and mostly likely to retrieve target documents for answering the question.\n"
        f"Note that each of your query should be an entity or a short phrase and its associated dimension.\n\n"
        f"Example Input:\n"
        f"Question: Are Marufabad and Nasamkhrali both located in the same country?\n"
        f"Example Output:\n"
        f"Query 1:\n"
        f"query_dimension: 'location_name'; query_content: 'Marufabad';\n"
        f"Query 2:\n"
        f"query_dimension: 'location_name'; query_content: 'Nasamkhrali';\n"
        f"Query 3:\n"
        f"query_dimension: 'location_type'; query_content: 'country';\n"
        f"Query 4:\n"
        f"query_dimension: 'location_relation'; query_content: 'located in';\n"
    )

    input_prompt = (
        f"Question: {query}"
    )
    
    class Query(BaseModel):
        query_content: str = Field(
            ...,
            title='Entity or phrase to query the documents'
        )
        query_dimension: Literal['location_name', 'location_type', 'location_relation'] = Field(
            ...,
            title='Dimension of the entity or phrase to query documents'
        )
        
    class AllQueries(BaseModel):
        list_of_queries: List[Query] = Field(
            ...,
            title='List of queries following the required format based on the question comprehension'
        )
        
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_prompt}
        ],
        max_tokens=4096,
        temperature=0,
        n=1,
        response_format=AllQueries,
    )

    try:
        detected_ents = response.choices[0].message.parsed
        
        if detected_ents is None or len(detected_ents.list_of_queries) == 0: return None
        cells = defaultdict(list)
        
        for ent in detected_ents.list_of_queries:
            cells[ent.query_dimension].append(ent.query_content)
            embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa_location.pkl', )
            # embed_string(target_str=ent.query_content, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa.pkl', )
        return cells
    
    except Exception as e:
        print(str(e))
        return None