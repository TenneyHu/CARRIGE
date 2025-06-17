from transformers import AutoModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from metrics_utils import calc_ingredients_diversity, calc_avg_semantic_diversity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the model to load')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load the model on')
    parser.add_argument('--results_file', type=str, default='./res/res1',
                        help='File path for retrieval results')
    parser.add_argument('--results_per_query', type=int, default=21,
        help='return results per query')
    parser.add_argument('--nums_query_rewriting', type=int, default=3,
        help='nums of query rewriting')
    return parser.parse_args()

def load_model(model_name, device):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

def calc_similarity(qid_to_texts, ground_truth):
    similarities = []
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
    min_score = -10.0
    max_score = 6.0
    for qid, texts in qid_to_texts.items():
        query = ground_truth[qid]
        pairs = [[query, doc] for doc in texts]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=128)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            for score in scores:
                score = (score - min_score) / (max_score - min_score)
                similarities.append(score)
    return sum(similarities) / len(similarities)

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_name, args.device)
    res = []

filelist = ["./logs/baseline_llama_0.7.out"]
qid_to_title = {}
qid_to_ingredients = {}
for file in filelist:
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                title = parts[1].split("Ingredientes")[0].lstrip("Nombre: ")
                title = title.lstrip("Nombre: ").rstrip()
                try:
                    ingredients = parts[1].split("Ingredientes")[1].lstrip("Ingredientes: ").split("Pasos")[0].rstrip()
                except:
                    ingredients = ""

                if qid not in qid_to_title:
                    qid_to_title[qid] = []
                    qid_to_ingredients[qid] = []
                qid_to_title[qid].append(title)
                qid_to_ingredients[qid].append(ingredients)

#print ("Exaples Titles: ",qid_to_title['0'])
#print ("Exaples Ingredients: ",qid_to_ingredients['0'])
ground_truth = {}
qid = 0
with open("./data/input.txt", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        title = parts[0].lstrip("Nombre: ")
        ground_truth[str(qid)] = title
        qid += 1

similarity_score = calc_similarity(qid_to_title, ground_truth)
ingredients_diversity_score = calc_ingredients_diversity(qid_to_ingredients)
diversity = calc_avg_semantic_diversity(model, qid_to_title)
print (f"{ingredients_diversity_score:.3f}", f"{diversity:.3f}", f"{float(similarity_score):.3f}")

