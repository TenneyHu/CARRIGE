from carrot_processing import load_recipe_adaption_query
import argparse
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--t', type=float, default=0.7,
                    help='Temperature for text generation')
parser.add_argument('--min_p', type=float, default=0,
                    help='Min_P for text generation')
parser.add_argument('--model', type=str, default='llama3.1',
                    help='Model name')

args = parser.parse_args()

def load_contexts(dir="./logs/ir_carrot_mmr_0.6.out", K=5):
    context = defaultdict(dict)
    with open(dir, encoding='utf-8') as f:
        for _, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue 
            qid = parts[0]
            context_string = "\t".join(parts[1:])
            if len(context[qid]) < K:
                context[qid][len(context[qid])] = context_string
    return context


template_query_context_history_str = (
    "Convierte la siguiente receta en una receta española para que se adapte a la cultura española, sea coherente con el conocimiento culinario español y se alinee con el estilo de las recetas españolas y la disponibilidad de ingredientes, y asegúrate de que sea diferente de las recetas en el historial proporcionado posteriormente. \n"
    "A continuación se muestran algunas recetas españolas relevantes recuperadas mediante búsqueda, que pueden ser útiles para la tarea:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada la receta original {query_str}, utiliza las recetas anteriores recuperadas para adaptarla a una receta española.\n"
    "A continuación se presentan algunos historiales; se debe EVITAR recomendar recetas similares a estas.\n"
    "---------------------\n"
    "{history_str}\n"
    "---------------------\n"
    "\n"
    "Instrucciones:\n"
    "Busca recetas relevantes entre aquellas marcadas con la etiqueta [reference] para usarlas como referencia. Evita seleccionar recetas que sean similares a las marcadas con la etiqueta [history]. \n"
    "La receta resultante debe estar completa, incluyendo ingredientes detallados e instrucciones paso a paso. Puedes guiarte por el estilo de las recetas españolas recuperadas.\n"
    "Da formato a tu respuesta exactamente de la siguiente manera:\n"
    "Nombre: [Título]\n"
    "Ingredientes: [Ingrediente 1] [Ingrediente 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "Por favor, empieza con \"Nombre: \" y no añadas ningún otro texto fuera de este formato.\n"
    "Mejor respuesta:"
)


queries = load_recipe_adaption_query("./data/input.txt")
qid = 0
llm = Ollama(model=args.model, request_timeout=300.0, temperature=args.t, min_p=args.min_p)

#context = load_contexts()
context = load_contexts("./logs/ir_carrot.out")
history_recipes = {}

for query in queries:
    for num in range(5):
        #try:
            
            index = str(qid)
            
            if qid in history_recipes:
                historys = "\n".join(f"[historial ] {recipe}" for recipe in history_recipes[qid])
            else:
                historys = ""

            carrot_prompt = PromptTemplate(template_query_context_history_str)
            req = carrot_prompt.format(context_str=context[index][num], query_str=query, history_str=historys)
            response = llm.complete(req)
            res = str(response).replace("\n", "\t")
            print(str(qid) + "\t" + res)
            recipe = str(response).replace('\n', ' ').split("Ingredientes")[0]
            if qid in history_recipes:
                history_recipes[qid].append(recipe)
            else:
                history_recipes[qid] = [recipe]
            
        #except Exception as e:
        #    print(f"Error processing query {qid}: {e}")
    qid += 1 
