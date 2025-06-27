
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
import re
from unidecode import unidecode
import inflect
import ast
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

p = inflect.engine()

def limpiar_especificaciones(ingredientes, palabras_eliminar):
    ingredientes_limpios = []
    for ingrediente in ingredientes:
        for palabra in palabras_eliminar:
            ingrediente = ingrediente.replace(palabra, "").strip()
        ingredientes_limpios.append(ingrediente)
    return ingredientes_limpios

def remove_empty_strings(lst):
    return [item for item in lst if item != '']

def remove_text_in_parentheses(text):
    # This regex matches any text between parentheses and the parentheses themselves
    result = re.sub(r'\(.*?\)', '', text)
    return result

def remove_text_after_coma(text):
    # remove text after coma
    result = text.split(',')[0]   
    return result

def remove_punctuation(text):
    # Remove punctuation using regex
    return re.sub(r'[^\w\s]', '', text)

def singularize_ingredient(ingredient):
    # Split the ingredient into words
    words = ingredient.split()
    # Singularize each word if it is plural
    singularized_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
    # Join the words back into a single string
    singularized_ingredient = ' '.join(singularized_words)
    return singularized_ingredient

def remove_everything_after(list_words, text):
    for word in list_words:
        if word in text:
            text = text.split(word)[0]
    return text

# Función para limpiar cada línea del texto
def limpiar_linea(linea,pattern):
    return re.sub(pattern, '', linea, flags=re.IGNORECASE).strip()

def clean_ingredients(recipes, ai_generated=False):

    remove_after_word = [' entre', ' al gusto', ' para', 'para ']
    palabras_eliminar = ["mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos", 
                     "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
                       " ()", "pequeño", "regulares", "lb. ", " suficiente"]
    ingredient_list = []
    
    if ai_generated:
        texto = recipes.lower() # lowercase
        texto = texto.replace("¼", "1/4")
        texto = texto.replace("½", "1/2")
        texto = texto.replace("¾", "3/4")
        texto = texto.replace("⅓", "1/3")
        texto = texto.replace("Al gusto,", "")
        texto = texto.replace("Al gusto:", "")
        ingredientes = [item.strip() for item in re.split(r'[-*]', texto) if item.strip()]
        ingredientes_principal = [i.split(',')[0] for i in ingredientes]
        ingredientes_principal = [i.split('(')[0] for i in ingredientes_principal]
        
    else:
        if recipes.strip().startswith("["):
            try:
                ingredientes = ast.literal_eval(recipes.strip())
                ingredientes = [ing.lower() for ing in ingredientes]
                ingredientes_principal = [remove_text_in_parentheses(ing) for ing in ingredientes]
                ingredientes_principal = [remove_text_after_coma(ing) for ing in ingredientes_principal]
                ingredientes_principal = [remove_everything_after(remove_after_word, ing) for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("¼", "1/4") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("½", "1/2") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("¾", "3/4") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("⅓", "1/3") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("cta.de,", "") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("cinco:", "5") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("azãºcar", "azúcar") for ing in ingredientes_principal]
            except:
                None
        else:
            texto = remove_text_in_parentheses(recipes)
            texto = remove_text_after_coma(texto)
            texto = remove_everything_after(remove_after_word, texto)
            texto = texto.lower()
            texto = texto.replace("¼", "1/4")
            texto = texto.replace("½", "1/2")
            texto = texto.replace("¾", "3/4")
            texto = texto.replace("⅓", "1/3")
            texto = texto.replace("al gusto,", "")
            texto = texto.replace("al gusto:", "")
            texto = texto.replace("cta.de", "")
            texto = texto.replace("cinco", "5")
            texto = texto.replace("azãºcar", "azúcar")
            ingredientes_principal = texto.split(',')
    
    pattern = r'\b\d+\s*(?:\/\d+)?\s*(?:g|gramos|pieza|centímetro cúbico|cucharada sopera|cc|un kilo|tableta|pisca|piezas|paquete|paquetes|un kilo|tarro|loncha|cabeza|cabezas|lonchas|kilogramo|kilogramos|puñados|chorrito|botella|gr.|libra|ralladura|lámina|barra|paquete|bastante|caja|rama|puñado|manojo|bote|vaso|pellizco|unidad|chorro|vaso|lata|rama|postre|litro|litros|mililitros|barra|-|cucharada colmada|unidades|copa|un kilo|kilo|gr|kg|ml|cl|dl|l|cm|bolas|dientes|diente|pizca|cucharadas soperas|tiras|tajadas|cucharaditas|cucharadita|cucharadas|mix|cucharada|cc|cda|cdta)?(?:\s+(?!de\b)(?:taza|tazas))?\b\s*(?:de)?'

    lineas_limpias = []
    lineas_limpias = [limpiar_linea(linea,pattern).lower() for linea in ingredientes_principal if linea.strip()]
    lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

    final_ingredients = []
    for i in lineas_limpias_final:
        if ' y ' in i:
            two_ingredientes = i.split(' y ')
            final_ingredients.append(two_ingredientes[0])
        elif 'o' in i:
            two_ingredientes = i.split(' o ')
            final_ingredients.append(two_ingredientes[0])
        else:
            final_ingredients.append(i)

    res = limpiar_especificaciones(final_ingredients, palabras_eliminar)
    res = [singularize_ingredient(ingredient).rstrip('.') for ingredient in res ]
    res  = [limpiar_linea(linea,pattern).lower().rstrip('.') for linea in res]
    res = remove_empty_strings(res)
    res = limpiar_especificaciones(res, ['cc de ','. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %', 'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ', ' en cuadradito', 'semilla de ', 'copita de '])
    # unidecode
    res= [unidecode(ing) for ing in res]
    ingredient_list.extend(res)
    return ingredient_list


def calc_avg_semantic_diversity(model, qid_to_texts):
    diversities = []
    for texts in qid_to_texts.values():
        if len(texts) < 2:
            continue  
        embeddings = model.encode(texts,show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
        upper_triangular = similarity_matrix[np.triu_indices(len(texts), k=1)]
        diversity = 1 - np.mean(upper_triangular)
        diversities.append(diversity)
    return np.mean(diversities) if diversities else 0

def calc_ingredients_diversity(qid_to_ingredients):
    diversities = []
    for texts in qid_to_ingredients.values():
        ingredients_set = set()
        all_ingredients = 0  
        for ingredients in texts: 
            ingredient_list = clean_ingredients(ingredients, ai_generated=True)
            all_ingredients += len(set(ingredient_list))
            ingredients_set.update(set(ingredient_list))
        if all_ingredients > 0:
            diversity_ingredient = 1.0 * len(ingredients_set) / all_ingredients
            diversities.append(diversity_ingredient)
    return np.mean(diversities) if diversities else 0

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
    similarities = [sim.cpu().numpy() for sim in similarities]
    similarities = np.array(similarities)
    return similarities.mean() if similarities.size > 0 else 0
    
    
