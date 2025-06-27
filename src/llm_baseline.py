from carrot_processing import load_recipe_adaption_query
import argparse
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

parser = argparse.ArgumentParser()

parser.add_argument('--t', type=float, default=1.2,
                    help='Temperature for text generation')
parser.add_argument('--min_p', type=float, default=0.05,
                    help='Min_P for text generation')
parser.add_argument('--model', type=str, default='llama3.1',
                    help='Model name')

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
llm = Ollama(model=args.model, request_timeout=300.0, temperature=args.t, min_p=args.min_p)

carrot_prompt = PromptTemplate(template_query_str)
for query in queries:
    for num in range(5):
        try:
            index = str(qid)
            req = carrot_prompt.format(query_str=query)
            response = llm.complete(req)
            res = str(response).replace("\n", "\t")
            print(str(qid) + "\t" + res)

        except Exception as e:
            print(f"Error processing query {qid}: {e}")
    qid += 1 

