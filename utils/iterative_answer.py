import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils.llm_calling import llm_answer

from utils.cube_router_wikimultihop import route_query_to_retrieve_docs



# Set up your API key and LLM
import os
os.environ["OPENAI_API_KEY"] = "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"


# Be sure to have your OPENAI_API_KEY set as an environment variable
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0) #gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20


def solve_multi_hop_query_iterative(model_type, original_query, corpus, k):
    print(f"\n >>> Original Query: '{original_query}'")
    
    # Use GPT-4o to generate the first subquery
    first_subquery_prompt = f"""
    Based on the query "{original_query}", what is the first, most direct and simplest single-hop question you need to ask to begin answering it? 
    For example the query: what is the nationality of the director of Wrong Turn 5: Bloodlines? You need to start with asking who is the director of Wrong Turn 5: Bloodlines?
    For example the query: What is the nationality of the director of Dark River (2017 Film)? You need to start with asking who is the director of Dark River (2017 Film)? 
    For example the query: where did Theodore Salisbury Woolsey's father study? You need to start with asking who is Theodore Salisbury Woolsey's father?
    For example the query: Who is the sibling-in-law of Jean Tangye? you can start from asking who is the spouse of Jean Tangye? and then asking who is the sibing of the spouse of Jean Tangye? as the second question.
    For example the query: Who is the paternal grandmother of Sigrid Of Sweden (1566–1633)? You can start from asking who is the father of Sigrid Of Sweden (1566–1633)? and then asking who is the mother of <ANSWER> of the above question? Then you can know the paternal grandmother of Sigrid Of Sweden (1566–1633). The paternal grandmother means the mother of the father of someone.
    For example, if the original query is asking who is younger, died earlier, etc. you need to start the first question with when?
    For example, if the query asks where was the place of death of Edsel Ford's father, you need to start the first question with who is Edsel Ford's father. 
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
            Based on all the information gathered: {intermediate_answers}, provide a final answer to the original query: "{original_query}".
            You must directly output the final answer without any other explanations.
            If the query asks the dates or locations, only output the specific dates and locations. 
            If the answer is a city, just output the city name and do not output the country it belongs to.
            If it is a yes-or-no query, only output yes or no.
            If the query asks who, only output the person name.
            If the query asks the comparison between two things, only output the one you think is correct without any other explainations..
            If the query ask nationality of someone, directly output the country name, e.g., Denmark or Frence. Do not output Danish or French.
            """
            final_answer = llm.invoke(final_prompt).content
            print(f"\n >>> Final Answer: {final_answer} \n")

            return final_answer, final_return_doc_idxs
        

        return_docs, return_doc_idxs = route_query_to_retrieve_docs(current_query, corpus, k)

        final_return_doc_idxs.append(return_doc_idxs)

        answer = llm_answer(model_type, current_query, return_docs)
        print(f"Intermediate Answer: {answer}")
        
        # Store the answer to pass it to the next step
        intermediate_answers[f"step_{step}"] = {"query": current_query, "answer": answer}
        
        # Use GPT-4o to determine the next action
        next_step_prompt = f"""
        Given the original query: "{original_query}"
        And the intermediate steps taken so far: {intermediate_answers}
        
        What is the next logical question to ask to continue solving the original query? 
        If the original query can be fully answered with the current information, you must state 'FINAL ANSWER' directly without any other explanation. 
        For example, if the original query asks who is the someone's paternal grandfather, once intermediate steps include someone's father's father, then you should state 'FINAL ANSWER' directly without any other explanation. 
        if the original query asks who is the someone's father-in-law, once intermediate steps include someone's spouse's father, then you should state 'FINAL ANSWER' directly without any other explanation. 
        if the original query asks who is the someone's child-in-law, once intermediate steps include someone's child's spouse, then you should state 'FINAL ANSWER' directly without any other explanation. 
        if the original query asks who is the someone's stepfather, once intermediate steps include someone's mother's spouse, then you should state 'FINAL ANSWER' directly without any other explanation. 
        if the original query asks who is the someone's stepmother, once intermediate steps include someone's father's spouse, then you should state 'FINAL ANSWER' directly without any other explanation.
        if the original query asks who is the someone's parenteral grandmother/grandfather, once intermediate steps include someone's father's mother/father, then you should state 'FINAL ANSWER' directly without any other explanation. 
        Otherwise, you must provide the next question only without any other explanations.
        """
        
        next_action = llm.invoke(next_step_prompt).content.strip()
        
        if next_action == "FINAL ANSWER" or current_query.startswith("FINAL ANSWER"):
            final_prompt = f"""
            Based on all the information gathered: {intermediate_answers}, provide a final answer to the original query: "{original_query}".
            You must output the final answer without any other explanations.
            If the query asks the dates or locations, only output the specific dates and locations. If the answer is a city, just output the city name and do not output the country it belongs to.
            If it is a yes-or-no query, only output yes or no.
            If the query asks who, only output the person name.
            If the query asks the comparison between two things, only output the one you think is correct without any other explainations.
            If the query ask nationality of someone, directly output the country name, e.g., Denmark. Do not output Danish.
            """
            final_answer = llm.invoke(final_prompt).content
            print(f"\n >>> Final Answer: {final_answer} \n")

            return final_answer, final_return_doc_idxs
        
        # Replace the current query with the next one
        current_query = next_action
        step += 1
        
    print("Process finished without a final answer.")
    return None



