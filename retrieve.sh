# python src/retrieve.py --reranking 0 --query_rewrite 0 > ./logs/ir_naive.out #naive ir
# python src/retrieve.py --reranking 1 --query_rewrite 1  > ./logs/ir_carrot.out
nohup python src/retrieve.py --reranking 1 --query_rewrite 1 --reranking_type "MMR" --reranking_alpha 0.2 > ./logs/ir_carrot_mmr_0.2.out 2>&1 &