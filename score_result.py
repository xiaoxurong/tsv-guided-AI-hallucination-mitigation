import json
import torch
import numpy as np
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from datasets import load_dataset
from tqdm import tqdm

def score_mitigation():
    # 1. Load the Results File 
    print("Loading results...")
    with open("mitigation_results.json", "r") as f:
        results = json.load(f)

    # 2. Load the TruthfulQA Dataset (to get the CORRECT answers)
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    
    # Map questions to correct answers for easy lookup
    # TruthfulQA usually provides multiple correct answers per question
    qa_map = {item['question']: item['correct_answers'] for item in dataset}

    # 3. Load the BLEURT Scorer (Same as tsv_main.py)
    print("Loading BLEURT Scorer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    scorer.eval()

    baseline_scores = []
    mitigated_scores = []

    print("Scoring responses...")
    for entry in tqdm(results):
        question = entry['question']
        baseline_ans = entry['baseline']
        mitigated_ans = entry['mitigated']
        
        # Get the list of ALL valid correct answers for this question
        correct_answers = qa_map.get(question, [])
        if not correct_answers: 
            continue

        # Helper function to get max score against any correct answer
        def get_score(prediction, references):
            with torch.no_grad():
                # Prepare inputs
                inputs = tokenizer([prediction] * len(references), references, padding='longest', return_tensors='pt')
                for k, v in inputs.items(): inputs[k] = v.to(device)
                
                # Get scores
                scores = scorer(**inputs).logits.flatten().tolist()
                return max(scores) # We take the best match

        # Score Baseline
        b_score = get_score(baseline_ans, correct_answers)
        baseline_scores.append(b_score)

        # Score Mitigated
        m_score = get_score(mitigated_ans, correct_answers)
        mitigated_scores.append(m_score)

    # 4. Calculate Final Stats
    avg_base = np.mean(baseline_scores)
    avg_mit = np.mean(mitigated_scores)
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Base Model Score:      {avg_base:.4f}")
    print(f"Mitigated Model Score: {avg_mit:.4f}")
    print(f"Improvement (Lift):    {((avg_mit - avg_base) / avg_base) * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    score_mitigation()