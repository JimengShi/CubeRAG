import os
import json
import argparse
from openai import OpenAI

# ================ Setup & Client ================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ================ Unified Prompt Logic ================
def get_unified_extraction_prompt(sentence):
    """
    Combines the extraction across all dimensions into a single-turn process.
    """
    dimensions_desc = """    
    1. cultural_product_media: cultural product names/phrases/terms related to TV series, film, song, magzine, book, award, opera, album, sport, competition, game, and bands/sports teams.
    2. cultural_product_type: type of cultural product names/phrases/terms, such as film, movie, song, magzine, book, award, opera, album, sport, match, competition, and game.
    3. action_relation: descriptive action/relation terms that describe a cultural product, such as "released on", "came out on", "directed by", "composed by", "awarded", "won the prize". 
    4. person_name: specific names of people, groups, such as historical figures, public figures (politicians, artists, scientists, athletes), fictional or semi-fictional characters, authors, directors, actors, musicians etc.
    5. role_occupation: descriptive role/occupation phrases/terms such as director, actor, musician, detective, star, player, commentator, captain, researcher etc.  
    6. person_behavior: descriptive behaviors of a person, such as die on, born on, birth, married to, study, work at, work as, death, located at, known for, was born, place of birth.
    7. nationality: descriptive phrases/terms on nationality, such as "come from", "is from", "nationality is" etc.
    8. relationship: social & personal relations description between person and person, company, location, organization, events, and others.
    9. date: temporal facts of events, such as release years, birth/death dates, occurred during, founding years, chronological ordering (“before/after”), and historical periods, such as Wars, Political eras, Cultural movements, Generational context. You need to include date/time information and the description of correspoinding objects. Example: "<film> released year is 1889", "John birthdate is Dec 8, 2012"
    10. location_name: specific natural locations (rivers, mountains, forest, trail, park, wetlands, ocean), geographic locations (restaurant, town, city, state, province, highway, church, airport) etc.
    11. location_relation: geospatial relations of geographical locations, such as "located in", "located at", "is a part of", "belongs to", "next to" etc.
    12. company_organization: names of company, headquarter, organizations, department, academic institute, religious groups, teams, groups, etc.
    13. politics: political events, rules, talks, acts, orders, political parties, government agencies, wars, weapons, etc.
    14. plant_animal: plant genera, plant species, ecosystem-specific plants (e.g., fen vegetation), such as Colocasia, Coronilla etc;  plant categories; animal species
    15. food: food names including but not limited to chocolate, fast-food, delivery etc.
    """

    prompt = f"""
    Extract entities from the sentence below based on these 15 dimensions:
    {dimensions_desc}. You can expand the dimensional values of each dimension if you think it is related.

    Guidelines:
    - Count the frequency of each entity found.
    - Use lowercases for the extracted terms.
    - For each dimension, feel free to include more entities/terms beyond the given examples.
    - Return a JSON object where keys are the dimension names and values are dictionaries of {{ "entity": frequency }}.
    - If a dimension has no entities, return an empty dictionary for that key.

    Sentence: "{sentence}"
    """
    return prompt

# ================ Main Processing ================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dimensional Entity Extraction")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., scifact)")
    parser.add_argument("--input", type=str, required=False, help="Path to input text file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define the dimensions we expect to see in the output
    dims = [
        "cultural_product_name", "cultural_product_type", "action_relation",
        "person_name", "role_occupation", "person_behavior", "nationality", "relationship",
        "date", "location_name", "location_relation", "company_organization",
        "politics", "plant_animal", "food_name"
    ]

    # Create output directory
    output_dir = f"hypercube_new/{args.data}"
    os.makedirs(output_dir, exist_ok=True)

    if args.data == "scifact":
        file_path = "corpus/scifact/pubmed_abstract.txt"
    elif args.data == "legalbench":
        file_path = "corpus/legalbench/contractnli.txt"
    elif args.data == "hurricane":
        file_path = "corpus/hurricane/SciDCC-Hurricane.txt"
    elif args.data == "hotpotqa":
        file_path = "corpus/hotpotqa/hotpotqa_corpus_with_index.json"

    # Read input lines of txt file
    # with open(file_path, "r") as f:
        # lines = [line.strip() for line in f if line.strip()] #
    
    # Read input lines of json file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    for sample in data:
        lines.append(sample['title'] + ': ' + sample['text'])
    
    for idx, line in enumerate(lines):
        print(f"Processing line {idx + 1}/{len(lines)}...")
        
        try:
            response = client.chat.completions.create(
                # model="gpt-4o-2024-08-06",
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": get_unified_extraction_prompt(line)}],
                response_format={"type": "json_object"}, # Ensures valid JSON output
                temperature=0,
            )

            # Parse the unified response
            full_result = json.loads(response.choices[0].message.content)

            # Distribute results into their respective dimension files
            for d in dims:
                # Get the specific dict for this dimension, default to empty if missing
                dim_data = full_result.get(d, {})
                
                output_file = f"{output_dir}/{d}.txt"
                with open(output_file, "a") as f_out:
                    f_out.write(json.dumps(dim_data) + "\n")

        except Exception as e:
            print(f"Error on line {idx+1}: {e}")
            # Log empty dicts for all dimensions on failure to keep line alignment
            for d in dims:
                with open(f"{output_dir}/{d}.txt", "a") as f_out:
                    f_out.write(json.dumps({}) + "\n")

    print(f"Extraction complete. Files saved in {output_dir}")

if __name__ == "__main__":
    main()