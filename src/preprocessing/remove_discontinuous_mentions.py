import os
import re

import pyprind

import utils

def main():
    for split in ["train", "dev", "test"]:
        process(split=split)

def process(split):
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_root = os.path.join(config["data"], "craft-conll", split)
    filenames = os.listdir(path_root)
    filenames = [n for n in filenames if n.endswith(".conll")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        lines = utils.read_lines(os.path.join(path_root, filename))
        with open(os.path.join(path_root, filename.replace(".conll", ".continuous_only_conll")), "w") as f:
            for line in lines:
                if line == "":
                    f.write("%s\n" % line)
                elif line.startswith("#"):
                    f.write("%s\n" % line)
                else:
                    items = line.split("\t")
                    assert len(items) >= 12
                    span_exps = items[-1]
                    span_exps = span_exps.split("|")
                    span_exps = [e for e in span_exps if not include_chars(e)]
                    if len(span_exps) > 0:
                        items[-1] = "|".join(span_exps)
                    else:
                        items[-1] = "-"
                    line = "\t".join(items)
                    f.write("%s\n" % line)

def include_chars(text):
    match = re.findall(r"([a-z]+)", text)
    return len(match) > 0

if __name__ == "__main__":
    main()

