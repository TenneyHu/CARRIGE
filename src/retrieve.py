import argparse
from carrot2 import CarrotRetriever, carrot_query_processing
from carrot_processing import load_data, load_recipe_adaption_query, recipe_split

def parse_args():
    parser = argparse.ArgumentParser(description='Recipe RAG System')
    parser.add_argument('--emb_model', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the embedding model to use')
    parser.add_argument('--index_dir', type=str, default='/data1/zjy/spanish_adaption_index/',
                        help='Directory to save/load the index')
    parser.add_argument('--save_index', type=int, default=0,
                        help='switch of saving index to the disk')
    parser.add_argument('--debugging', type=int, default=0,
                        help='display rettrieve logs')
    parser.add_argument('--query_rewrite', type=int, default=1,
                        help='switch of rewriting')
    parser.add_argument('--reranking', type=int, default=1,
                        help='switch of reranking')
    parser.add_argument('--reranking_type', type=str, default="relevance",
                        help='type of reranking')
    parser.add_argument('--reranking_alpha', type=float, default=0.7,
                        help='contorl of diversity in reranking')
    parser.add_argument('--input_file_dir', type=str, default="./data/input.txt",
                        help='input file dir')
    parser.add_argument('--k_per_querytype', type=int, default=5,
                        help='retrieval cut off')
    parser.add_argument('--final_k', type=int, default=5,
                        help='retrieval cut off')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    titles, full_document_maps = load_data(filter_county="ESP")
    #print(f"Loaded {len(titles)} docs")
    
    carrot_retriever = CarrotRetriever(
        emb_model=args.emb_model,
        index_dir=args.index_dir,
        full_document_maps=full_document_maps,
        debugging = args.debugging,
        reranking = args.reranking,
        reranking_type = args.reranking_type,
        reranking_alpha = args.reranking_alpha,
        res_num_per_query = args.k_per_querytype,
        final_res_num = args.final_k
    )

    if args.save_index: 
        carrot_retriever.save_index(titles)
    carrot_retriever.load_index()
    queries = load_recipe_adaption_query(args.input_file_dir)
    qid = 0

for query in queries:
    title, ingredients, steps = recipe_split(query)
    content = "Ingredientes:\t" + ingredients + "\t" + "Pasos:\t" + steps
    transformed_queries = carrot_query_processing(title, content, args.debugging, args.query_rewrite)
    contexts = carrot_retriever.retrieve(transformed_queries)
    for n in contexts:
        contexts = n.full_text.replace("\n","\t")
        print(str(qid) + "\t" + contexts) 


    qid += 1
