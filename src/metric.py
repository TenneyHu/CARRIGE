import argparse
from transformers import AutoModel
from metrics_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./logs/llm_gemma2_9b_temp0.7.out',
                        help='Input')
    parser.add_argument('--model_name', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the model to load')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to load the model on')
    parser.add_argument('--generated_result', type=int, default=1,
                        help='1=llm, 0=ir')

    return parser.parse_args()

def load_model(model_name, device):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_name, args.device)
    res = []

    qid_to_title = {}
    qid_to_ingredients = {}
    qid_to_contents = {}
    qid_to_recipe = {}
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if args.generated_result:
                if len(parts) >= 2:
                    qid = parts[0]
                    title = parts[1].split("Ingredientes")[0].lstrip("Nombre: ")
                    title = title.lstrip("Nombre: ").rstrip()
                    content = " ".join(parts[1:])

                parts = line.strip().split()
                ingredients = ""
                collecting = False
                for x in parts:
                    if x.strip().startswith("Ingredientes"):
                        collecting = True
                        continue
                    elif x.strip().startswith("Pasos"):
                        break
                    if collecting:
                        ingredients += x + " "
            else:
                if len(parts) >= 3:
                    qid = parts[0]
                    title = parts[2]
                    title = title.lstrip("Nombre: ").rstrip()
                    content = " ".join(parts[1:])
                else:
                    continue
                
                ingredients = ""
                collecting = False
                for x in parts:
                    if x.strip().startswith("Ingredientes"):
                        collecting = True
                        continue
                    elif x.strip().startswith("Pasos"):
                        break
                    if collecting:
                        ingredients += x + " "
                ingredients = re.sub(r"[\'\"\[\]]", "", ingredients).strip()

            if qid not in qid_to_title:
                qid_to_title[qid] = []
                qid_to_ingredients[qid] = []
                qid_to_contents[qid] = []
                qid_to_recipe [qid]= []

            qid_to_title[qid].append(title)
            qid_to_ingredients[qid].append(ingredients)
            qid_to_recipe[qid].append("Nombre: " + title + " Ingredientes: " + ingredients)
            qid_to_contents[qid].append(content)


    #print ("Exaples Titles: ",qid_to_recipe['0'])
    #print ("Exaples Ingredients: ",qid_to_ingredients['0'])
    #print ("Exaples Contents: ",  qid_to_contents['0'])
    
    ground_truth = {}
    qid = 0
    with open("./data/input.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            title = parts[0].lstrip("Nombre: ")
            ground_truth[str(qid)] = "Nombre: " + title + "Ingredientes: " + parts[1]
            qid += 1

    print (qid_to_ingredients['0'])
    perinput_ingredients_diversity = calc_perinput_ingredients_diversity(qid_to_ingredients)
    acrossinput_ingredients_diversity = calc_acrossinput_ingredients_diversity(qid_to_ingredients)
    semantic_diversity = calc_avg_semantic_diversity(model, qid_to_title)
    #lexical_diversity = compute_global_avg_uniq_n(qid_to_contents)
    lexical_diversity = compute_per_input_avg_uniq_n(qid_to_contents)
    cas = calc_cas(qid_to_recipe)
    bertscore = calc_bertscore_similarity(qid_to_recipe, ground_truth)

    '''
    print(
        f"Input Name: {args.input}\n"
        f"Perinput Ingredients Diversity: {perinput_ingredients_diversity:.3f}\n"
        f"Perinput Semantic Diversity: {semantic_diversity:.3f}\n"
        f"Acrossinput Lexical Diversity: {lexical_diversity:.3f}\n"
        f"Acrossinput Ingredients Diversity: {acrossinput_ingredients_diversity:.3f}\n"
        f"CAS: {cas:.3f}\n"
        f"BertScore: {bertscore:.3f}\n")
    '''
    filename = args.input.split("/")[-1]
    print(
        filename, "&",
        f"{lexical_diversity:.3f}", "&",
        f"{perinput_ingredients_diversity:.3f}", "&",
        f"{semantic_diversity:.3f}", "&",
        #f"{acrossinput_ingredients_diversity:.3f}", "&",
        f"{cas:.3f}", "&",
        f"{bertscore:.3f} \\\\" ,
    )