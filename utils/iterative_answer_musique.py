import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils.llm_calling import llm_answer


from utils.cube_router_musique import route_query_to_retrieve_docs_musique

from utils.merge_return_doc import merge_return_doc_idx
from utils.load_data import load_qa, load_corpus

corpus_data_path = 'corpus/musique/musique_corpus_with_index.json'
corpus = load_corpus(corpus_data_path)



# Set up your API key and LLM
import os
os.environ["OPENAI_API_KEY"] = "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"


# Be sure to have your OPENAI_API_KEY set as an environment variable
llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0) #gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20




def solve_multi_hop_query_iterative_musique(model_type, original_query, corpus, k):
    print(f"\n >>> Original Query: '{original_query}'")
    
    # Use GPT-4o to generate the first subquery
    first_subquery_prompt = f"""
    Given a multi-hop query, you MUST decompose it to several one-hop simple subquery.
    Based on a query "{original_query}", what is the first, most direct and simplest single-hop question you need to ask to begin answering it? You MUST begin with the simple ONE-HOP question in logics as simple as possible. Do NOT include attributive clause in the question.
    You can not output question like "What is the capital of the state where Charles Mingus was born?" since it is two hops. You must ask where Charles Mingus was born first, then ask What is the capital of the state.
    For example, if the original query: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona? you need to start asking who is the person Messi's goals in Copa del Rey compared to get? then based on the answer, you ask when was the person signed by Barcelona.
    If the original query asks, when was the start of the battle of the birthplace of the performer of III? you need to start asking who is the performer of III? then where is the birthplace the the performer? and when did the battle of birthplace name?
    For example, if the original query is asking who is younger, died earlier, etc. you need to start the first question with when?
    For example, if the original query asks where was the place of death of Edsel Ford's father, you need to start the first question with who is Edsel Ford's father. 
    For example, if the original query asks Who was in charge of the place where Castricum is located? you need to start the first question with where is Castricum is located.
    For example, if the orginal query asks What city is the star of Sous les pieds des femmes from? you need to start the first question with who is the star of Sous les pieds des femmes, and then ask what city is the start from.
    For example, if the original query is How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?, you need to figure out the country of the person Lim and then figure out the boundary lies between thailand and the answer of above question.
    Provide only the question.
    """
    current_query = llm.invoke(first_subquery_prompt).content.strip().strip('"').strip("'")
    
    intermediate_answers = {}
    final_return_doc_idxs = []
    step = 1
    
    while current_query:
        print(f"\nStep {step}: Generating subquery.")
        print(f"Current Subquery: {current_query}")

        if current_query == "FINAL ANSWER" or current_query.startswith("FINAL ANSWER") or step > 4:
            final_prompt = f"""
            Based on all the information gathered: {intermediate_answers}, retrieved documents {[corpus[i] for i in merge_return_doc_idx(final_return_doc_idxs, 5)]}, provide a final answer to the original query: "{original_query}".
            If the query ask what/which/who/when/where/how many/what year, you must directly output the final answer as precise as possible without any other explanations. For example, "Beijing" or "300", or "director", central Atlantic Ocean.
            If the query asks when, you should output the time or data as precise as possible, e.g., mid-June, or June 1982.
            If the query asks the dates or locations, only output the specific dates and locations. If the answer is a city, just output the city name and do not output the country it belongs to.
            If it is a yes-or-no query, only output yes or no.
            If the query asks the comparison between two things, e.g., what came out first, you only output the name directly without any explanations.
            """
            final_answer = llm.invoke(final_prompt).content
            print(f"\n >>> Final Answer: {final_answer} \n")

            return final_answer, final_return_doc_idxs
        

        return_docs, return_doc_idxs = route_query_to_retrieve_docs_musique(current_query, corpus, k)

        final_return_doc_idxs.append(return_doc_idxs)

        answer = llm_answer(model_type, current_query, return_docs)
        print(f"Intermediate Answer: {answer}")
        
        # Store the answer to pass it to the next step
        intermediate_answers[f"step_{step}"] = {"query": current_query, "answer": answer}
        
        # Use GPT-4o to determine the next action
        next_step_prompt = f"""
        Given the original query: "{original_query}"
        And the intermediate steps taken so far: {intermediate_answers}
        
        What is the next logical question to ask to continue solving the original query? You could output next logical question with one-hop as simple as possible. Do Not include attributive clause in the question.
        If the original query can be fully answered with the current information, you must state 'FINAL ANSWER' directly without any other explanation. 
        For example, if the original query asks who is the someone's paternal grandfather, once intermediate steps include someone's father's father, then you should state 'FINAL ANSWER' directly without any other explanation. 
        Otherwise, you must provide the next question only without any other explanations.
        """
        
        next_action = llm.invoke(next_step_prompt).content.strip()
        
        if next_action == "FINAL ANSWER" or current_query.startswith("FINAL ANSWER"):
            final_prompt = f"""
            Based on all the information gathered: {intermediate_answers}, retrieved documents {[corpus[i] for i in merge_return_doc_idx(final_return_doc_idxs, 5)]}, provide a final answer to the original query: "{original_query}".
            If the query ask what/which/who/when/where/how many/what year, you must directly output the final answer as precise as possible without any other explanations. For example, "Beijing" or "300", or "director"".
            If the query asks when, you should output the time or data as precise as possible, e.g., mid-June, or June 1982.
            If the query asks the dates or locations, only output the specific dates and locations. If the answer is a city, just output the city name and do not output the country it belongs to.
            If it is a yes-or-no query, only output yes or no.
            If the query asks the comparison between two things, e.g., what came out first, you only output the name directly without any explanations.
            """
            final_answer = llm.invoke(final_prompt).content
            print(f"\n >>> Final Answer: {final_answer} \n")

            return final_answer, final_return_doc_idxs
        
        # Replace the current query with the next one
        current_query = next_action
        step += 1
        
    print("Process finished without a final answer.")
    return None