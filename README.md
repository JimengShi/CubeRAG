# MultiCube-RAG

<!-- <div align="left">
<img src="https://github.com/JimengShi/Hypercube-RAG/blob/main/figures/hypercube_rag.jpg" alt="hypercuberag" width="1000"/> 
</div> -->


### Project Structure

- `QA`: saves the question-answering pairs
- `corpus`: saves the original corpus
- `hypercube`: constructs and saves hypercube
- `evaluation`: computes evaluation scores
- `utils`: helper functions
- `qa_rag_wikimultihop.py`: script to run MultiCube-RAG for wikimultihop dataset


### Environment
```
conda create --name multicube python==3.10
conda activate multicube

pip install -r requirements.txt
```

### Quick start on wikimultihop

For example, the following script takes the wikimultihop dataset and GPT-4o-mini as an LLM base.

```
python qa_rag_wikimultihop.py --data wikimultihop --model gpt-4o-mini --retriever hypercube --save true
```

Parameter descriptions:

- `--data`: dataset name
- `--model`: llm name
- `--retriever`: retriever method
- `--save`: if saving the final output or not

### MultiCube-RAG Example
<div align="left">
<img src="https://github.com/JimengShi/CubeRAG/blob/main/assets/multicube_example.png" alt="multicube-rag" width="1000"/> 
</div>
