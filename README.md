

# GrAlgoBench

Large Reasoning Models (LRMs) have achieved rapid progress, yet existing benchmarks—focused primarily on mathematics, programming, or common-sense reasoning—remain limited by **poor difficulty control**, **ambiguous evaluation**, and a **narrow coverage of reasoning paradigms**.  

**GrAlgoBench** introduces a new benchmark centered on **graph algorithm problems** to evaluate the reasoning ability of LRMs. Compared with prior benchmarks, graph tasks offer several unique advantages:  

- **Fine-grained reasoning**: emphasize step-by-step logical execution.  
- **Scalable difficulty control**: adjustable by graph size and topology.  
- **Standardized evaluation**: objective, programmatic correctness checks.  
- **Rich reasoning paradigms**: covering enumeration, exploration, and heuristic decision-making.  

Through experiments on **nine tasks across three categories**, we reveal critical weaknesses of current LRMs:  

1. **Poor intuitive reasoning** – models struggle with heuristic-based tasks.  
2. **Execution errors** – frequent mistakes in step-by-step algorithm execution.  
3. **Limited memory** – difficulty recalling nodes, edges, and intermediate states.  
4. **Over-thinking** – excessive but ineffective self-verification attempts.  

Together, these findings highlight **graph algorithm problems** as a **rigorous, multidimensional, and application-relevant testbed**, exposing the limitations of today’s LRMs and guiding future progress in reasoning research.  

<p align="center">

<img src="overview.png" alt="GrAlgoBench Overview" width="600">

</p>
---

## 📂 Project Structure

```
GrAlgoBench/
├── data_generation/        # Scripts for dataset construction
├── Inference/              # Model inference scripts and configs
├── error_analysis/         # Scripts for analyzing model errors
├── logs/                   # Default log directory
├── results/                # Saved inference results
├── scripts/                # Helper bash scripts for batch running
└── README.md               # Documentation
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda create -n gralgobench python=3.10
conda activate gralgobench
pip install -r requirements.txt
```

### 2. Data Generation
```
python ./data_generation/build_dataset.py
```

Datasets are organized as:
```
dataset/
├── MKC_easy.pkl
├── MKC_medium.pkl
├── MST_hard.pkl
└── ...
```

### 3. Run Inference
Single-task example:
```bash
python Inference/infer_open.py \
    --LLM Qwen3-8B \
    --task MST \
    --difficulty medium \
    --batch_size 32 \
    --gpu_num 4
```

Batch execution via script:
```bash
bash scripts/infer_open.sh
```


### 4. 🔍 Error Analysis

To analyze model errors, follow these steps:

1. **Reformat raw responses**  
   This step parses and normalizes model outputs into a consistent structure:  

   ```bash
   python error_analysis/reformat.py
   ```

2. **Run the error analysis script**  
   Specify the task, evaluation model, and the model that generated the responses:  

   ```bash
   python error_analysis/error_analysis.py \
       --task MST \
       --llm gpt5_mini \
       --response_generated_from_what_model Qwen3-8B
   ```

---



