import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline

def smooth_curve(x, y, num=300):
    x = np.array(x)
    y = np.array(y)

    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)

    x_smooth = np.linspace(x.min(), x.max(), num)
    y_smooth = poly(x_smooth)

    return x_smooth, y_smooth


def curve():

    llama_x = [0.464, 0.444, 0.404, 0.361]
    llama_y = [0.042, 0.137, 0.232, 0.293]
    llama_labels = ['t=0.1', 't=0.4', 't=0.7', 't=1.0']

    carrige_y = [0.183, 0.203, 0.269, 0.314]
    carrige_x = [0.467, 0.459, 0.442, 0.422]
    carrige_labels = ['t=0.1', 't=0.4', 't=0.7', 't=1.0']

    qwen_y = [0.065, 0.170, 0.247, 0.307]
    qwen_x = [0.471, 0.460, 0.439, 0.416]
    qwen_labels = ['t=0.1', 't=0.4', 't=0.7', 't=1.0']

    llama_culture_y = [0.413, 0.415, 0.451, 0.481]
    carrige_culture_y = [0.413, 0.431, 0.463, 0.499]
    qwen_culture_y = [0.388, 0.395, 0.404, 0.429]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=True)

    for ax, metric_y, ylabel, title in [
        (axes[0], [llama_y, qwen_y, carrige_y], "Diversity (Semantic Diversity Score)", "Preservation vs Diversity"),
        (axes[1], [llama_culture_y, qwen_culture_y, carrige_culture_y], "CultureScore", "Preservation vs Cultural Appropriateness")
    ]:
        for x, y, labels, color, marker, name in [
            (llama_x, metric_y[0], llama_labels, 'red', 's', 'Closed-book LLaMA3.1'),
            (qwen_x, metric_y[1], qwen_labels, 'blue', 's', 'Closed-book Qwen2.5'),
            (carrige_x, metric_y[2], carrige_labels, 'magenta', 's', 'CARRIGE-LLaMA'),
        ]:
            x_smooth, y_smooth = smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, '--', color=color, label=name)
            ax.plot(x, y, marker=marker, linestyle='', color=color)
            for xi, yi, label in zip(x, y, labels):
                ax.scatter(xi, yi, color=color, marker=marker, s=80, edgecolor='black', zorder=5)
                ax.text(xi, yi, label, color=color, fontsize=9)

        ax.set_xlabel("Preservation (BERTScore)", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("./figures/trade-off.png")
    plt.show()

def curve_plus():
    # First group: CARROT-MMR IR (λ)
    mmr_labels_1 = ['λ=0.2', 'λ=0.4', 'λ=0.6', 'λ=0.8']
    mmr_x_1 = [0.259, 0.289, 0.298, 0.301]
    mmr_y_1 = [0.645, 0.602, 0.527, 0.489]
    mmr_culture_y_1 = [0.657, 0.519, 0.503, 0.501]

    # Second group: CARROT-MMR RAG (temp)
    mmr_labels_2 = ['t=0.1', 't=0.4', 't=0.7', 't=1.0']
    mmr_x_2 = [0.586, 0.575, 0.545, 0.508]
    mmr_y_2 = [0.038, 0.099, 0.164, 0.206]
    mmr_culture_y_2 = [0.306, 0.343, 0.393, 0.434]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # First row: CARROT-MMR IR
    x_smooth, y_smooth = smooth_curve(mmr_x_1, mmr_y_1)
    axes[0, 0].plot(x_smooth, y_smooth, '--', color='green', label='CARROT-MMR IR')
    axes[0, 0].plot(mmr_x_1, mmr_y_1, 'o', color='green')
    for xi, yi, label in zip(mmr_x_1, mmr_y_1, mmr_labels_1):
        axes[0, 0].scatter(xi, yi, color='green', edgecolor='black', s=80, zorder=5)
        axes[0, 0].text(xi, yi, label, color='green', fontsize=9)
    axes[0, 0].set_xlabel("Preservation (BERTScore)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Diversity (Semantic Diversity Score)", fontsize=12, fontweight='bold')
    axes[0, 0].set_title("Preservation vs Diversity (MMR λ)", fontsize=13, fontweight='bold')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    x_smooth_c, y_smooth_c = smooth_curve(mmr_x_1, mmr_culture_y_1)
    axes[0, 1].plot(x_smooth_c, y_smooth_c, '--', color='green', label='CARROT-MMR IR')
    axes[0, 1].plot(mmr_x_1, mmr_culture_y_1, 'o', color='green')
    for xi, yi, label in zip(mmr_x_1, mmr_culture_y_1, mmr_labels_1):
        axes[0, 1].scatter(xi, yi, color='green', edgecolor='black', s=80, zorder=5)
        axes[0, 1].text(xi, yi, label, color='green', fontsize=9)
    axes[0, 1].set_xlabel("Preservation (BERTScore)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("CultureScore", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Preservation vs Cultural Appropriateness (MMR λ)", fontsize=13, fontweight='bold')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Second row: CARROT-MMR RAG
    x_smooth, y_smooth = smooth_curve(mmr_x_2, mmr_y_2)
    axes[1, 0].plot(x_smooth, y_smooth, '--', color='orange', label='CARROT-MMR RAG')
    axes[1, 0].plot(mmr_x_2, mmr_y_2, 'o', color='orange')
    for xi, yi, label in zip(mmr_x_2, mmr_y_2, mmr_labels_2):
        axes[1, 0].scatter(xi, yi, color='orange', edgecolor='black', s=80, zorder=5)
        axes[1, 0].text(xi, yi, label, color='orange', fontsize=9)
    axes[1, 0].set_xlabel("Preservation (BERTScore)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Diversity (Semantic Diversity Score)", fontsize=12, fontweight='bold')
    axes[1, 0].set_title("Preservation vs Diversity (Temperature)", fontsize=13, fontweight='bold')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    x_smooth_c, y_smooth_c = smooth_curve(mmr_x_2, mmr_culture_y_2)
    axes[1, 1].plot(x_smooth_c, y_smooth_c, '--', color='orange', label='CARROT-MMR RAG')
    axes[1, 1].plot(mmr_x_2, mmr_culture_y_2, 'o', color='orange')
    for xi, yi, label in zip(mmr_x_2, mmr_culture_y_2, mmr_labels_2):
        axes[1, 1].scatter(xi, yi, color='orange', edgecolor='black', s=80, zorder=5)
        axes[1, 1].text(xi, yi, label, color='orange', fontsize=9)
    axes[1, 1].set_xlabel("Preservation (BERTScore)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("CultureScore", fontsize=12, fontweight='bold')
    axes[1, 1].set_title("Preservation vs Cultural Appropriateness (Temperature)", fontsize=13, fontweight='bold')
    axes[1, 1].grid(True)
    axes[1, 1].legend()


    plt.tight_layout()
    plt.savefig("./figures/tradeoff_plus1.png")
    plt.show()

def col():
    with open("./res", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data = [[float(x.strip()) for x in line.strip().replace('\\\\', '').split('&')[1:]] for line in lines]
    data_array = np.array(data)
    correlation_matrix_np = np.corrcoef(data, rowvar=False)
    labels = [
        "Lexical Diversity",
        "Ingredient Diversity", 
        "Semantic Diversity", 
        "CultureScore", 
        "BERTScore"
    ]


    wrapped_labels = [
        "Lexical \n Diversity", 
        "Ingredient \n Diversity", 
        "Semantic \n Diversity", 
        "CultureScore", 
        "BERTScore"
    ]

    correlation_df_labeled = pd.DataFrame(correlation_matrix_np, index=labels, columns=labels)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        correlation_df_labeled, annot=True, cmap="coolwarm",
        center=0, fmt=".2f", square=True, xticklabels=wrapped_labels, yticklabels=wrapped_labels
    )
    plt.xticks(rotation=45, ha='right')  
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("./figures/correlation.png")



curve()
curve_plus()