import os
import json
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pickle
from utils.topk_most_freq import topk_most_freq_id
from utils.decompose_hotpotqa import decompose_query_along_company_organization


# from IPython import embed
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
    

# # ================ read corpus ================
# def read_corpus():
#     with open('corpus/hotpotqa/hotpotqa_corpus_with_index.json', 'r', encoding='utf-8') as f:
#         data = json.load(f)

#         corpus = []
#         for sample in data:
#             corpus.append(sample['title'] + ': ' + sample['text'])

#     # return corpus[:4000]
#     return corpus


# ================ construct hypercube ================
print('Reading the hypercube files ...')


dimensions = ['company_organization', 'location_name', 'person_name', 'date']
hypercube = {'company_organization': defaultdict(list),
             'location_name': defaultdict(list),
             'person_name': defaultdict(list),
             'action_relation': defaultdict(list),
             'date': defaultdict(list)}

model = SentenceTransformer('intfloat/e5-base-v2')

for dimension in dimensions:
    with open(f'hypercube_new/hotpotqa/{dimension}.txt') as f:
        readin = f.readlines()
        # readin = readin[:10]
        for i, line in tqdm(enumerate(readin), total=len(readin), desc=f"{dimension}"):
            tmp = json.loads(line)
            for k in tmp:
                hypercube[dimension][k].append(i)
                
                # embed_string(target_str=k, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa_company_organization.pkl', )
                embed_string(target_str=k, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa.pkl')


# ================ hypercube retriever ================
def get_docs_from_cells(cells, top_k):
    if cells is None: return []
    tmp_ids = []
    for key, v in cells.items():        
        for vv in v:
            vv = vv.lower()
            if vv in hypercube[key]:
                tmp_ids.extend(hypercube[key][vv])
            else:
                # embed_string(target_str=vv, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa_location.pkl')
                embed_string(target_str=vv, emb_model=model, dict_path='ent2emb/ent2emb_hotpotqa.pkl')
                # vv_emb = ent2emb[vv]

                vv_emb = model.encode(vv, normalize_embeddings=True)

                for cand in hypercube[key]:
                    if ent2emb[cand] @ vv_emb > 0.9: 
                        tmp_ids.extend(hypercube[key][cand])

    doc_ids = topk_most_freq_id(tmp_ids, top_k)

    return list(doc_ids)


def retriever_hypercube_company_organization(query, corpus, k):
    # corpus = read_corpus()
    
    cells = decompose_query_along_company_organization(query, dimensions)
    print(f">>> Identified cells from the query: {cells} \n")

    doc_ids = get_docs_from_cells(cells, k)
    print(f'>>> Doc ids by hypercube retrieval: {[id + 1 for id in doc_ids]} \n')    

    docs = '\n\n'.join([f"Document {idx + 1}: {corpus[doc_id]}" for idx, doc_id in enumerate(doc_ids)])

    return docs, [id + 1 for id in doc_ids]


