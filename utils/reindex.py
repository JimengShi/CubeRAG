import json 
import argparse



def load_dataset(data_path):
    with open(data_path, "r") as f:
        return json.load(f)



def main():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    args = parser.parse_args()

    qa_list = []
    qa_list_old = load_dataset(args.data_path)  #json_file = 'QA/hurricane/synthetic_qa.json'
    # print(qa_list)

    for i, sample in enumerate(qa_list_old):
        qa_list.append({
            "index": i+1,
            "question": sample['question'],
            "answer": sample['answer'],
            
        })

    with open(args.data_path, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, indent=2, ensure_ascii=False)

    print(f"Reindex {len(qa_list)} Q&A pairs to {args.data_path}")



if __name__ == "__main__":
    main()


