from together import Together
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def llm_answer(model_type, query, return_docs):
    
    # set up instruction
    instruction = 'Answer the query based on the given retrieved documents. Please keep the answer as precise as possible. \
        If the query ask what/which/who/when/where/how many/what year, you must directly output the final answer as precise as possible without any other explanations. For example, "Beijing" or "300", or "director", central Atlantic Ocean. \
            If the query asks when, you should output the time or data as precise as possible, e.g., mid-June, or June 1982. \
            If the query asks the dates or locations, only output the specific dates and locations. If the answer is a city, just output the city name and do not output the country it belongs to. \
            If it is a yes-or-no query, only output yes or no. \
            If the query asks the comparison between two things, e.g., what came out first, you only output the name directly without any explanations. \
    If you find the output include two or more answers, e.g., The film "Flags and Waves" was created by animators Bill Reeves and Alain Fournier. You need to output all. If related information of both cannot be available, you need to ouput either one whose information is available. \
    If you find the output include two or more answers, e.g., Who is the performer of the song "Dip". You need to output Tyga and Nicki Minaj. If related information of both cannot be available, you need to ouput either one whose information is available. \
    The retrieved documents are:\n'

    # set up the LLM
    if model_type == 'llama-70B-Instruct-Turbo':
        os.environ["TOGETHER_API_KEY"] = ""  # or set it externally
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + return_docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ],
            temperature=0
        )
        res = completion.choices[0].message.content  
        return res


    elif model_type == 'qwen-7B-Instruct-Turbo':
        os.environ["TOGETHER_API_KEY"] = ""  # or set it externally
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo", 
            messages=[
                {"role": "system", "content": instruction + return_docs},
                {"role": "user", "content": 'Query: ' + query + '\nAnswer:'}
            ],
            temperature=0
        )
        res = completion.choices[0].message.content  
        return res
    
    
    elif model_type == 'gpt-4o-mini':
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": instruction + return_docs},
                {
                    "role": "user",
                    "content": 'Query: ' + query + '\nAnswer:'
                }
            ],
            temperature=0
        )
        res = completion.choices[0].message.content
        return res