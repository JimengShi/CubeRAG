import openai
import json
from collections import defaultdict
import json
from openai import OpenAI
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


client = OpenAI()

def read_file(file_path):
    # with open(file_path, "r") as f:
        # lines = f.readlines()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        corpus = []
        for sample in data:
            corpus.append(sample['title'] + sample['text'])
        
    return corpus



def prompt_template_dim(dim):
    if dim == "cultural_product_name":
        prompt_template = """
        Extract cultural product entities/phrases related to film, movie, song, magzine from the following sentence and count their frequency. 
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"Bratuku Teruvu": 1, "Apprentice To Murder": 1, "Days and Hours": 1}. 
        If some entities appear more than once, for example, "Days and Hours" appear twice, "Apprentice To Murder" appears three times, 
        then count the total number of appearances, and output {"Bratuku Teruvu": 1, "Apprentice To Murder": 3, "Days and Hours": 2}. 
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "action_relation":
        prompt_template = """
        Extract all action/relation terms that describe film, movie, song, magzine from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"released": 1, "came out": 1, "directed by": 1, "composed by": 1, "director": 1, "performer": 1}. 
        If some entities appear more than once, for example, "released" appear twice, "directed by" appears three times, 
        then count the total number of appearances, and output {"released": 2, "came out": 1, "directed by": 3, "composed by": 1, "director": 1, "performer": 1}. 
        Please feel free to add other actions and relations if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "cultural_product_type":
        prompt_template = """
        Identify and extract the type of the cultural products (film, movie, song, magzine) from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related type and values are their frequency.
        You should directly return the results, Example Output: {"movie": 1, "film": 1, "song": 1, "magzine": 1}. 
        If some entities appear more than once, for example, "movie" appear twice, "film" appears three times, then count the total number of appearances, and output {"movie": 2, "film": 3, "song": 1, "magzine": 1}.
        If the type does not appear in the sentence, just skip it.
        Please feel free to add more types of a cultural product if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "person":
        prompt_template = """
        Extract person names from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"Hanro Smitsman": 1, "Peter Levin": 1, "Jason Moore": 1}. 
        If some entities appear more than once, for example, "Hanro Smitsman" appear twice, "Jason Moore" appears three times, 
        then count the total number of appearances, and output {"Hanro Smitsman": 2, "Peter Levin": 1, "Jason Moore": 3}. 
        Please feel free to add other person names if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "role":
        prompt_template = """
        Extract roles a person can play from the following sentence and count their frequency, such as director, spouse, child, mother, father, grandfather, performer, composer, father-in-law, stepmother, mother-in-law, stepmother, child-in-law, sibling-in-law, husband, wife, uncle, founder, paternal grandmother, maternal grandfather.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"director": 1, "paternal grandmother": 1, "spouse": 1, "grandfather": 1, "composer": 1}. 
        If some entities appear more than once, for example, "director" appear twice, "spouse" appears three times, 
        then count the total number of appearances, and output {"director": 2, "paternal grandmother": 1, "spouse": 3, "grandfather": 1, "composer": 1}. 
        Please feel free to add other roles if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "behavior":
        prompt_template = """
        Extract behaviors of a person from the following sentence and count their frequency, such as die on, born on, birth, married to, study, work at, work as, death, located at.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"die on": 1, "married to": 1, "spouse": 1, "work at": 1, "death": 1, "located at": 1}. 
        If some entities appear more than once, for example, "die on" appear twice, "work at" appears three times, 
        then count the total number of appearances, and output {"die on": 2, "married to": 1, "work at": 3, "death": 1, "located at": 1}.
        Please feel free to add other behaviors if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "nationality":
        prompt_template = """
        Extract nationality descriptions/phrases of a person from the following sentence and count their frequency, such as come from, is from, the country is.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"come from": 1, "is from": 1, "the country is": 1}. 
        If some entities appear more than once, for example, "come from" appear twice, "is from" appears three times, 
        then count the total number of appearances, and output {"come from": 2, "is from": 3, "work at": 3, "the country is": 1}.
        Please feel free to add more nationality descriptions/phrases of a person if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "location_name":
        prompt_template = """
        Extract location entities/descriptions/phrases from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related entities/concepts and values are their frequency.
        You should directly return the results, Example Output: {"Marufabad": 1, "Nasamkhrali": 1, "Florida": 1, "USA": 1}. 
        If some entities appear more than once, for example, "Marufabad" appear twice, "Nasamkhrali" appears three times, 
        then count the total number of appearances, and output {"Marufabad": 2, "Nasamkhrali": 3, "work at": 3, "Florida": 1, "USA": 1}.
        Please feel free to add more location_name descriptions/phrases of a person if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "location_type":
        prompt_template = """
        Identify and extract the type of the specific locations from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related type and values are their frequency.
        You should directly return the results, Example Output: {"street": 1, "village": 1, "city": 1, "county": 1, "state": 1, "country": 1, "school": 1, "university": 1, "company": 1}. 
        If some entities appear more than once, for example, "city" appear twice, "county" appears three times, 
        then count the total number of appearances, and output {"street": 1, "village": 1, "city": 2, "county": 3, "state": 1, "country": 1, "country": 1, "school": 1, "university": 1, "company": 1}.
        If the location type does not appear in the sentence, just skip it.
        Please feel free to add more types of the location if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """

    elif dim == "location_relation":
        prompt_template = """
        Identify and extract the relations of the specific locations from the following sentence and count their frequency.
        Return the result as a dictionary where keys are related type and values are their frequency.
        You should directly return the results, Example Output: {"located at": 1, "is a part of": 1, "belongs to": 1, "based on": 1, "founded by": 1, "near": 1, "designed by": 1, "named after": 1}. 
        If some entities appear more than once, for example, "located at" appear twice, "is a part of" appears three times, 
        then count the total number of appearances, and output {"located at": 2, "is a part of": 3, "belongs to": 1, "based on": 1, "founded by": 1, "near": 1, "designed by": 1, "named after": 1}.
        Please feel free to add more terms that describe relations of differet locations if you find any.
        Do not change the lowercases and uppercases of extracted entities/phrases/terms included in the sentences.
        You must output the result with a dictionary format only. Do not include any extra formatting or explanation.
        Sentence is:
        """


    return prompt_template




# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--dim", type=str, required=True, help="Path to the dimensions.")
    return parser.parse_args()



# ================ Main Function ================
def main():
    args = parse_args()

    if args.data == "scifact":
        file_path = "corpus/scifact/pubmed_abstract.txt"
    elif args.data == "wikimultihop":
        file_path = "corpus/wikimultihop/2wikimultihopqa_corpus_with_index.json"
    
    lines = read_file(file_path)
    prompt_template = prompt_template_dim(args.dim)

    output_dir = f"hypercube/{args.data}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, line in enumerate(lines):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt_template + line}], # + line["title"] + line["text"]
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            print(f">>> content {idx+1}: {content}")

            result_dict = eval(content)  # or use json.loads if JSON format is enforced
            # results.append(result_dict)

            # Save result to file line by line (append mode)            
            with open(f"{output_dir}/{args.dim}.txt", "a") as f:
                f.write(json.dumps(result_dict) + "\n")

        except Exception as e:
            print(f"Error on line {idx+1}: {e}")
            # Optionally write an empty dict for failed lines
            with open(f"{output_dir}/{args.dim}.txt", "a") as f:
                f.write(json.dumps({}) + "\n")


    print("Done!")


if __name__ == "__main__":
    main()

