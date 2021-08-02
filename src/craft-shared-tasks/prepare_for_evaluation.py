import argparse
import os
import re

import utils


BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")


def main(args):
    path_gold = args.gold
    path_pred = args.pred

    dictionary, gold_documents = get_from_gold(path_gold)
    pred_documents = get_from_pred(path_pred)
    print("Number of gold documents: %d" % len(gold_documents))
    print("Number of pred documents: %d" % len(pred_documents))
    write_documents(gold_documents, pred_documents, dictionary, os.path.join(path_pred, "files-to-evaluate"))


def get_from_gold(path_gold):
    filenames = os.listdir(path_gold)
    dictionary = {} # doc_key -> filename
    documents = {} # doc_key -> lines (conll content)
    for filename in filenames:
        lines = utils.read_lines(os.path.join(path_gold, filename))

        begin_match = re.match(BEGIN_DOCUMENT_REGEX, lines[0])
        assert begin_match is not None
        doc_key = begin_match.group(1)
        assert not filename in dictionary
        dictionary[doc_key] = filename

        # documents[doc_key] = [l for l in lines if l != ""]
        documents[doc_key] = lines

    return dictionary, documents


def get_from_pred(path_pred):
    filenames = os.listdir(path_pred)
    filenames = [n for n in filenames if n.endswith(".conll")]
    assert len(filenames) == 1
    filename = filenames[0]

    documents = {} # doc_key -> lines (conll content)
    lines = utils.read_lines(os.path.join(path_pred, filename))
    doc_key = None
    for line in lines:
        if line.startswith("#begin document"):
            assert doc_key is None
            # New doc_key
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            assert begin_match is not None
            doc_key = begin_match.group(1)
            # Append
            documents[doc_key] = [line]
        elif line.startswith("#end document"):
            assert doc_key is not None
            doc_key = None
        elif line != "":
            assert doc_key is not None
            documents[doc_key].append(line)
    return documents


def write_documents(gold_documents, pred_documents, dictionary, output_dict):
    utils.mkdir(output_dict)
    for doc_key in pred_documents.keys():
        filename = dictionary[doc_key]
        gold_doc = gold_documents[doc_key]
        pred_doc = pred_documents[doc_key]
        assert len(gold_doc) >= len(pred_doc), (len(gold_doc), len(pred_doc))
        pred_i = 0
        with open(os.path.join(output_dict, filename), "w") as f:
            for gold_line in gold_doc:
                if gold_line.startswith("#begin document"):
                    pred_line = pred_doc[pred_i]
                    assert pred_line.startswith("#begin document")
                    f.write("%s\n" % gold_line)
                    pred_i += 1
                elif gold_line == "":
                    f.write("\n")
                else:
                    pred_line = pred_doc[pred_i]
                    assert gold_line != "" and pred_line != ""
                    gold_items = gold_line.split("\t")
                    pred_items = pred_line.split()
                    assert len(gold_items) == len(pred_items), (len(gold_items), len(pred_items))
                    new_line = gold_items[:-1] + pred_items[-1:]
                    new_line = "\t".join(new_line)
                    f.write("%s\n" % new_line)
                    pred_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

