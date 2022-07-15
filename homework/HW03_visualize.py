import json

from matplotlib import pyplot as plt

# load data from json

colors = {"micro_f1": "blue", "macro_f1": "red", "micro_map": "green", "macro_map": "orange"}


def plot_metrics(name):
    path = f"output/{name}_transform_metrics.json"
    with open(path) as f:
        data = json.load(f)

    # plot data
    plt.figure(figsize=(10, 10))
    for metric in ["micro_f1", "macro_f1", "micro_map", "macro_map"]:
        for type in ["train", "val"]:
            values = data[type][metric]
            plt.plot(values, label=f"{type} {metric}",
                     linestyle="--" if type == "train" else "-",
                     c=colors[metric])
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"'{name}'-transformation Metrics")
    plt.show()


for name in ["basic", "my", "no"]:
    plot_metrics(name)