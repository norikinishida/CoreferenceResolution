import argparse

import numpy as np
import tabulate

import utils


def safe_float(value):
    if isinstance(value, str):
        if "/" in value:
            l, r = value.split("/")
            return float(l) / float(r)
        else:
            return float(value)
    else:
        return float(value)


def main(args):
    df = utils.read_csv(path=args.path, delimiter="\t", with_head=True, with_id=True)
    list_of_dict = []
    for metric_type in ["bcub", "blanc", "ceafe", "ceafm", "lea", "muc"]:
        dictionary = {"metric": metric_type}
        dictionary.update({f"{x}-{y}": safe_float(df["TOTAL":][f":{metric_type}-{x}-{y}"].values[0]) for x in ["mention", "coref"] for y in ["p", "r", "f"]})
        list_of_dict.append(dictionary)
    dictionary = {"metric": "average"}
    dictionary.update({f"{x}-{y}": np.mean([d[f"{x}-{y}"] for d in list_of_dict]) for x in ["mention", "coref"] for y in ["p", "r", "f"]})
    list_of_dict.append(dictionary)
    print(tabulate.tabulate(list_of_dict, headers="keys", tablefmt="github", stralign="center", numalign="left"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

