from carrot_processing import load_recipe_adaption_query
import transformers
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--t', type=float, default=0.7,
                    help='Temperature for text generation')
parser.add_argument('--minp', type=float, default=0.1,
                    help='Min_P for text generation')
args = parser.parse_args()


template_query_str = (
    "Convierte la siguiente receta en una receta española para que se adapte a la cultura española, sea coherente con el conocimiento culinario español y se alinee con el estilo de las recetas españolas y la disponibilidad de ingredientes.\n"
    
    "Dada la receta original {query_str}, utiliza las recetas anteriores recuperadas para adaptarla a una receta española.\n"
    "\n"
    "Instrucciones:\n"
    "La receta resultante debe estar completa, incluyendo ingredientes detallados e instrucciones paso a paso. Puedes guiarte por el estilo de las recetas españolas recuperadas.\n"
    "Da formato a tu respuesta exactamente de la siguiente manera:\n"
    "Nombre: [Título]\n"
    "Ingredientes: [Ingrediente 1] [Ingrediente 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "Por favor, empieza con \"Nombre: \" y no añadas ningún otro texto fuera de este formato."
    "Mejor respuesta:"
)

queries = load_recipe_adaption_query("./data/input.txt")

qid = 0

model_dir = "/data2/huggingface-mirror/dataroot/models/meta-llama/Meta-Llama-3.1-8B-Instruct/"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:3",
)

for query in queries:
    for num in range(5):
        prompt = [
            {"role": "user", "content": template_query_str.format(query_str=query)}
        ]
        outputs = pipeline(
            prompt,
            max_new_tokens=512,
            min_p=0.1,
        )
        response = outputs[0]["generated_text"][-1]['content']
        res = response.replace("\n", "\t")
        print(str(qid) + "\t" + res)
    qid += 1
