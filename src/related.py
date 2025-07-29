from collections import defaultdict
from carrot_processing import load_recipe_adaption_query
from transformers import AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

def load_contexts(dir="./logs/ir_carrot_mmr_0.6.out", K=5, id = 2):
    context = defaultdict(dict)
    with open(dir, encoding='utf-8') as f:
        for _, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue 
            qid = parts[0]
            context_string = parts[id]
            if id == 1:
                context_string = context_string.lstrip("Nombre: ")
            if len(context[qid]) < K:
                context[qid][len(context[qid])] = context_string
    return context

ground_truth = {}
qid = 0
with open("./data/input.txt", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        title = parts[0].lstrip("Nombre: ")
        ground_truth[str(qid)] = title 
        qid += 1

context = load_contexts()
res = load_contexts("./logs/rag_naive.out", id = 1)
#res = load_contexts("./logs/CARRIGE_llama3.1_latest_temp1.0.out", id = 1)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True).to('cuda')
quries = load_recipe_adaption_query("./data/input.txt")
uniq_counts = Counter()
for q in tqdm(range(500)):
    qid = str(q)

    context_texts = list(context[qid].values())  # List of tensors
    res_texts = list(res[qid].values())
    #context_texts.append(ground_truth[qid])
    context_emb = model.encode(context_texts, show_progress_bar=False, device='cuda')  # shape: [N1, D]
    res_emb = model.encode(res_texts, show_progress_bar=False, device='cuda')          # shape: [N2, D]
    sim_matrix = cosine_similarity(res_emb, context_emb) 
    
    max_similarities = np.max(sim_matrix, axis=1)
    max_ids = np.argmax(sim_matrix, axis=1)
    id_counts = Counter()
    id_counts.update(max_ids)
    '''
    for idx, score in zip(max_ids, max_similarities):
        if score < 0.5:
            id_counts["no_match"] += 1
        else:
    '''

    num_unique_ids = len(id_counts)
    uniq_counts[num_unique_ids] += 1

print (uniq_counts)
