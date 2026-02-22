# MultiCube-RAG

We introduce a `multi-dimensional cube structure` to concisely and comprehensively represent flat text, facilitating the subsequent retrieval.
In our work, the core idea to address multi-hop questions is `divide and conquer` by decomposing the complex multi-hop query into multiple simpler one-hop subqueries.
Each subquery can flexibly select the most suitable cube-based retriever to fetch the most relevant knowledge from an external database, facilitating iterative reasoning and retrieval. 

## MultiCube-RAG for Iterative Reasoning and Retrieval
<div align="center">
<img src="https://github.com/JimengShi/CubeRAG/blob/main/assets/iterative_multicube_rag.png" alt="iterative-multicube-rag" width="500"/> 
</div>


## Project Structure

- `QA`: saves the question-answering pairs
- `corpus`: saves the original corpus
- `hypercube`: constructs and saves hypercube
- `gpt_extraction`: extract entities along hypercube dimensions
- `evaluation`: computes evaluation scores
- `utils`: helper functions
- `qa_rag_wikimultihop.py`: script to run MultiCube-RAG for wikimultihop dataset


## Environment
```
conda create --name multicube python==3.10
conda activate multicube

pip install -r requirements.txt
```

## Quick start on wikimultihop

### Step 1: Set up API key
```
export OPENAI_API_KEY="sk-"
```

### Step 2: Construct cube by extracting entities
For example, the following script takes the hotpot dataset and GPT-4o-mini as an LLM base.

```
cd gpt_extraction
python joint_extract_hotpot.py
```

### Step 3: Cube-based QA
```bash
CUDA_VISIBLE_DEVICES=0

python run_cube_rag.py --data hotpotqa --model gpt-4o-mini --retriever hypercube --save true
```

Parameter descriptions:

- `--data`: dataset name
- `--model`: llm name
- `--retriever`: retriever method
- `--save`: if saving the final output or not


## MultiCube-RAG Example
<div align="left">
<img src="https://github.com/JimengShi/CubeRAG/blob/main/assets/multicube_example.png" alt="multicube-rag" width="1000"/> 
</div>


