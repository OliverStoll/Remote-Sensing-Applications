import json

from matplotlib import pyplot as plt

# load data from json

colors = {"micro_f1": "blue", "macro_f1": "red", "micro_map": "green", "macro_map": "orange"}


def plot_metrics(name):
    path = f"output/{name}_transform_metrics.json"
    with open(path) as f:
        data = json.load(f)

    # plot data
    plt.figure(figsize=(7, 7))
    # reduce margins
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.07)
    for metric in ["micro_f1", "macro_f1", "micro_map", "macro_map"]:
        for type in ["train", "val"]:
            values = data[type][metric]
            plt.plot(values, label=f"{metric}" if type == "val" else None,
                     linestyle="--" if type == "train" else "-",
                     c=colors[metric])

    # add white line to legend
    plt.plot([], [], "w", label=" ")
    # add black dotted line to legend
    plt.plot([], [], "k--", label="train")
    plt.plot([], [], "k-", label="val")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"'{name}'-transformation Metrics")
    plt.show()


for name in ["no", "basic", "my"]:
    plot_metrics(name)
