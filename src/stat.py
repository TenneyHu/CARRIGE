from carrot_processing import load_recipe_adaption_query
import numpy as np
from metrics_utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_results(input_dir="../logs/ir_carrot_mmr_0.6.out", generated_result=False):
    qid_to_ingredients = {}
    with open(input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if generated_result:
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

            if qid not in qid_to_ingredients:
                qid_to_ingredients[qid] = []

            qid_to_ingredients[qid].append(ingredients)
    return qid_to_ingredients



def plot_combined_figure(counter_results, labels=None, top_k=50):
    model_names = ['Source', 'CARROT-MMR', 'CARRIGE', 'LLaMA']
    across_ing_diversity = [0.74, 0.65, 0.49, 0.47]
    normalized_ing_counts = [371, 315, 251, 234]

    # 创建上下两个子图的布局，宽度减半
    fig = plt.figure(figsize=(12, 6))  # 高度从10减少到8，让图表更扁
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    font_properties = {'fontsize': 12, 'fontweight': 'bold'} 

    # 上图：配料频率排名图
    ax1 = fig.add_subplot(gs[0])
    for i, counter_result in enumerate(counter_results):
        sorted_counts = sorted([int(count) for _, count in counter_result[:top_k]], reverse=True)
        if not sorted_counts:
            continue
        ranks = list(range(1, len(sorted_counts) + 1))
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        ax1.plot(ranks, sorted_counts, label=label, linewidth=2)

    ax1.set_xlabel('Rank', **font_properties)
    ax1.set_ylabel('Ingredient\nFrequency', **font_properties)
    ax1.grid(True)
    ax1.legend(prop={'size': 10, 'weight': 'bold'}, loc='upper right', frameon=False) 

    # 下图：双y轴柱状图
    ax2 = fig.add_subplot(gs[1])
    ax2_twin = ax2.twinx()  # 创建双y轴
    
    # 设置颜色
    colors1 = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']  
    
    # 计算柱子位置：左边4个柱子，右边4个柱子，中间留空
    left_positions = [0, 1, 2, 3]  # 左边4个位置
    right_positions = [4.5, 5.5, 6.5, 7.5]  # 右边4个位置，中间空隙减小
    
    # 左y轴：Across-Input Diversity (左边4个柱子)
    bars1 = ax2.bar(left_positions, across_ing_diversity, 
                    width=0.8, edgecolor='black', color=colors1, alpha=1.0)
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 右y轴：Normalized Ingredient Counts (右边4个柱子)
    bars2 = ax2_twin.bar(right_positions, normalized_ing_counts, 
                         width=0.8, edgecolor='black', color=colors1, alpha=1.0)
    for bar in bars2:
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height}', 
                      ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 设置x轴
    ax2.set_xlim(-0.5, 8)
    ax2.set_xticks(left_positions + right_positions)
    # 处理x轴标签重叠问题，使用换行
    x_labels = []
    for name in model_names:
        if name == 'CARROT-MMR':
            x_labels.append('CARROT-\nMMR')
        elif name == 'CARRIGE':
            x_labels.append('CARRIGE')
        else:
            x_labels.append(name)
    ax2.set_xticklabels(x_labels + x_labels, fontsize=10, fontweight='bold')
    
    # 添加中间的虚线分隔
    ax2.axvline(x=3.75, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # 设置左y轴（Across-Input Diversity）
    ax2.set_ylim(0.4, 0.8)
    ax2.set_ylabel('Across-Input\nDiversity', **font_properties)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 设置右y轴（Normalized Ingredient Counts）
    ax2_twin.set_ylim(200, 400)
    ax2_twin.set_ylabel('Normalized\nIngredient Counts', **font_properties)

    plt.tight_layout()
    fig.savefig("../figures/ingredients.png", dpi=300, bbox_inches='tight')
 

    
queries = load_recipe_adaption_query("../data/input.txt")
query_dict = {}
qid = 0
for query in queries:
    ing =  query.split("\t")[1] 
    ing = re.sub(r"[\'\"\[\]]", "", ing).strip()
    query_dict[qid] =[ing]
    qid += 1

labels =[
    "Source Recipes",
    "CARROT-MMR",
    "LLaMA 3.1",
    "CARRIGE-LLaMA"
]
counters = []
#print (calc_acrossinput_ingredients_diversity(query_dict, K=1))

avg, res_ori = count_ingredients(query_dict)
res_ori =  [(k, v) for k, v in res_ori if len(k.strip()) > 3]
counters.append(res_ori)
#plot_ingredient_rank_line(res_ori)



avg, res_ir = count_ingredients(load_results(input_dir="../logs/ir_carrot_mmr_0.6.out"))
res_ir =  [(k, v) for k, v in res_ir if len(k.strip()) > 3]
counters.append(res_ir)


avg, res_llm = count_ingredients(load_results(input_dir="../logs/llm_llama3.1_latest_temp0.7.out", generated_result=True))
res_llm =  [(k, v) for k, v in res_llm if len(k.strip()) > 3]
counters.append(res_llm)



avg, res_rag = count_ingredients(load_results(input_dir="../logs/CARRIGE1_llama3.1_latest_temp0.7.out", generated_result=True))
res_rag =  [(k, v) for k, v in res_rag if len(k.strip()) > 3]
counters.append(res_rag)

plot_combined_figure(counters,labels)

