import argparse
import os
import json

from utils.llm_calling import llm_answer
from utils.metric import semantic_score, f1_score, em_score
from utils.metric import recall_at_k
from utils.load_data import load_qa, load_corpus
from utils.iterative_answer import solve_multi_hop_query_iterative
from utils.merge_return_doc import merge_return_doc_idx



# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    # parser.add_argument("--data", type=str, required=True, help="Path to the QA dataset (JSON or CSV).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini",  "llama-70B-Instruct", "qwen-7B-Instruct"], help="Select llm to get answer.")
    parser.add_argument("--k", type=int, default=2, help="Number of retrieved documents for each iteration.")
    parser.add_argument("--topk", type=int, default=5, help="Number of retrieved documents for all iterations.")
    parser.add_argument("--save", type=str, default="false", choices=["true", "false"], help="Evaluation metric: one score or multiple scores.")
    return parser.parse_args()


# ================ Main Function ================
def main():
    args = parse_args()

    corpus_data_path = 'corpus/wikimultihop/2wikimultihopqa_corpus_with_index.json'
    qa_data_path = 'QA/wikimultihop/2wikimultihopqa_with_index.json'

    corpus = load_corpus(corpus_data_path)
    qa_samples = load_qa(qa_data_path)

    all_llm_outputs = []
    for i, sample in enumerate(qa_samples):
    
        original_query = sample['question']
        gold_answer = sample['answer']
        relevant_doc_idxs = sample['supporting_facts_index']
            
        print(f"************************ Original query {i+1}: {original_query} ************************")

        predicted_answer, returned_doc_idxs = solve_multi_hop_query_iterative(args.model, original_query, corpus, args.k)
        returned_doc_idxs = merge_return_doc_idx(returned_doc_idxs, args.topk)

        print(f">>> Predicted: {predicted_answer}")
        print(f">>> Ground Truth: {gold_answer}")
        print(f'Returned_doc_idxs: {returned_doc_idxs}')
        print(f'Relevant_doc_idxs: {relevant_doc_idxs}')
        print(f"em_score: {em_score(predicted_answer, gold_answer):.4f}")
        print(f"f1_score: {f1_score(predicted_answer, gold_answer):.4f}")
        print(f'Semantic_score: {semantic_score(predicted_answer, gold_answer):.4f} ')
        print(f'Recall_score: {recall_at_k(returned_doc_idxs, relevant_doc_idxs):.4f} \n')

        # Save output for each sample
        llm_output = {
            "index": i+1,
            "question": original_query,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "em_score": em_score(predicted_answer, gold_answer),
            "f1_score": f1_score(predicted_answer, gold_answer),
            "semantic_score": semantic_score(predicted_answer, gold_answer),
            "return_doc_idxs": returned_doc_idxs,
            "relevant_doc_idxs": relevant_doc_idxs,
            "recall": recall_at_k(returned_doc_idxs, relevant_doc_idxs)
        }

        output_dir = f"output/wikimultihop/"
        output_file_path = f"{output_dir}/llm_output_{args.model}_k{args.topk}_responses.json"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if args.save == "true":
            # Read existing data if the file already exists
            if os.path.exists(output_file_path):
                with open(output_file_path, "r", encoding="utf-8") as f:
                    try:
                        all_llm_outputs = json.load(f)
                    except json.JSONDecodeError:
                        all_llm_outputs = []   # Handle case where file is empty or corrupted

            # Append the new output dictionary
            all_llm_outputs.append(llm_output)

            # Write the entire updated list back to the file, overwriting the old content
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(all_llm_outputs, f, indent=4, ensure_ascii=False)


    # compute average scores
    num_samples = len(all_llm_outputs)
    total_semantic, total_f1, total_em, total_recall = 0, 0, 0, 0

    for entry in all_llm_outputs:
        total_em += entry.get("em_score", 0)
        total_f1 += entry.get("f1_score", 0)
        total_semantic += entry.get("semantic_score", 0)
        total_recall += entry.get("recall", 0)

    average_scores = {
        "total_samples": num_samples,
        "average_em_score": total_em / num_samples,
        "average_f1_score": total_f1 / num_samples,
        "average_semantic_score": total_semantic / num_samples,
        "average_recall_score": total_recall / num_samples
    }

    with open(f"{output_dir}/llm_output_{args.model}_k{args.topk}_scores.json", "w", encoding="utf-8") as f:
        json.dump(average_scores, f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()

