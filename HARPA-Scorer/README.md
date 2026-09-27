# RM-R1 Data Processing & Reward Model Training

This directory contains all scripts used to **prepare, analyze, and train** the HARPA reward model following the **RM-R1 / RLVR** framework. The pipeline is modular—each script in `code/` performs transformation in the data flow.

---

## 🧩 Overview

The HARPA scorer is trained to assess proposal *feasibility* and *execution success* using reasoning traces distilled from ASD agent executions (CodeScientist).  
The pipeline includes the following scripts:

1. **Execution trace extraction**
2. **Source classification and proposal pairing**
3. **Preference pair construction**
4. **Oracle reasoning trace generation**
5. **Distillation and RLVR reward model training**

---

## 📁 Directory Layout

```shell
code/
│
├── exec_trace_extraction_v2.py # Extract execution trace & proposal details to JSON
├── source_classification.py # Classify source papers & build extra proposal pairs
├── prep_preference_data_v2.py # Create preference data for reward modeling
├── trace_generation_data.py # Prepare input data for oracle reasoning generation
├── gen_anthropic_trace_data_v2.py # Generate oracle reasoning traces (teacher data)
├── analyse_trace_gen.py # Analyze oracle generation quality / accuracy
├── SFT_dataset_v1.py # Build SFT dataset for distillation
├── SFT_dataset_rlvr_train.py # Build sft dataset ofr RLVR stage (without reasoning traces)
├── test_data.ipynb # Prepare validation/test splits for RM-R1
└── README.md # This file
```

`datafiles/` : Contains the reasoning data generated using the Claude-4 model, stored under `smoke_test_results/`, based on the constructed preference data pairs.
It also includes intermediate output files produced by the data preparation and processing scripts in the `code/` directory.

`results`: Contains the final evaluation outputs of the HARPA-Scorer on the test and validation datasets



---

### Training the Reward Model (RM-R1)

Once the data is ready, train the reward model using the RM-R1 and RLVR pipelines.

##### Distillation

```bash
conda run -n rm-r1-sft bash rm_r1/scripts/Distill/local/distill_qwen2.5-7b-instruct.sh
```

##### RLVR Fine-tuning

```bash
conda run -n rm-r1 bash rm_r1/scripts/RLVR/local/train_rm_r1_rlvr_qwen2.5_instruct_7b.sh
```
