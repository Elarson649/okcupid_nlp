import seaborn as sns
import matplotlib.pyplot as plt


def bar_graph(data, title, saveas, reverse=False):
    """
    Creates a horizontal bar graph. Usually used with unique_terms to display the terms the are most unique to a category.
    :param data: A series with the features as the index and the f-score as the value. Typically the product of unique_terms.
    :param title: The title of the bar graph.
    :param saveas: What to save the file as (e.g. 'graph.png')
    :param reverse: Should we reverse the color palette (default False)
    :return: Nothing
    """

    sns.axes_style("white")
    palette = sns.hls_palette(10, l=.6)
    if reverse:
        palette.reverse()
    sns.barplot(x=data.values, y=data.index, palette=palette)
    sns.despine()
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.title(title, size=16, fontweight='bold')
    plt.savefig(saveas, bbox_inches='tight')