import pandas as pd
import matplotlib.pyplot as plt


def visualize():
    data = pd.read_csv('Data/wholetrain_clean.tsv', sep='\t')
    print(data.head)
    strategy_count = data.groupby('strategy').count()
    plt.bar(strategy_count.index.values, strategy_count["reframed_text"])
    plt.xticks(rotation='vertical')
    plt.xlabel('Reframe Strategy')
    plt.ylabel('Number of Texts')
    plt.show()
    return strategy_count


visualize()
