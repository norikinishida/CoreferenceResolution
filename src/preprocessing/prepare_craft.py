from collections import defaultdict
import unicodedata
import os
import re

import utils

CONLL_KEYS = ["_0", "_1", "token-index", "token", "pos", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "span-exp"]

def main():
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_conll = os.path.join(config["craft"], "coref-conll")
    path_bionlp = os.path.join(config["craft"], "coref-bionlp")
    path_dst = os.path.join(config["data"], "craft")

    utils.mkdir(os.path.join(path_dst, "train"))
    utils.mkdir(os.path.join(path_dst, "dev"))
    utils.mkdir(os.path.join(path_dst, "test"))
    utils.mkdir(os.path.join(path_dst + "-conll", "train"))
    utils.mkdir(os.path.join(path_dst + "-conll", "dev"))
    utils.mkdir(os.path.join(path_dst + "-conll", "test"))

    # 辞書 (CoNLLファイル名 -> training/dev/test split) の作成
    path_split = os.path.join(config["craft"], "articles", "ids")
    train_filenames = utils.read_lines(os.path.join(path_split, "craft-ids-train.txt"))
    dev_filenames = utils.read_lines(os.path.join(path_split, "craft-ids-dev.txt"))
    test_filenames = utils.read_lines(os.path.join(path_split, "craft-ids-test.txt"))
    assert len(train_filenames) == 60
    assert len(dev_filenames) == 7
    assert len(test_filenames) == 30
    filename_to_split = {}
    for fname in train_filenames:
        filename_to_split[fname] = "train"
    for fname in dev_filenames:
        filename_to_split[fname] = "dev"
    for fname in test_filenames:
        filename_to_split[fname] = "test"

    # CoNLLファイルリスト
    filenames = os.listdir(path_conll)
    filenames = [n for n in filenames if n.endswith(".conll")]
    filenames.sort()

    n_doc = len(filenames)
    n_sents = 0
    n_chains = 0
    n_mensions = 0
    for filename in filenames:
        split = filename_to_split[filename.replace(".conll", "")]

        # CoNLLデータ読み込み
        sentences_conll = utils.read_conll(os.path.join(path_conll, filename), CONLL_KEYS)
        for s_i in range(len(sentences_conll)):
            for w_i in range(len(sentences_conll[s_i])):
                token = sentences_conll[s_i][w_i]["token"]
                # トークン正規化
                token = utils.normalize_string(token, able=["space", "hyphen", "amp", "quot", "lt", "gt"])
                token = token.strip()
                # もしCoNLLデータにて、一つの単語が空白分割で複数個のパーツに別れるなら、それらを"-"で1単語に結合しておく
                # (後のコードで空白でトークン分割するために)
                if len(token.split()) > 1:
                    old_token = token
                    token = "-".join(token.split())
                    utils.writelog("Transformed %s -> %s" % (old_token, token))
                sentences_conll[s_i][w_i]["token"] = token

        # BioNLPデータ読み込み
        inner_section_boundaries, outer_section_boundaries = read_section_boundaries_from_bionlp(os.path.join(path_bionlp, filename.replace(".conll", ".bionlp")))

        # CoNLLデータとBioNLPデータで文数が一致するかチェック
        # i.e., すべての文がいずれかのセクションに含まれているか
        assert len(sentences_conll) \
                == sum([(end_i - begin_i + 1) for (begin_i, end_i) in inner_section_boundaries]) \
                == sum([(end_i - begin_i + 1) for (begin_i, end_i) in outer_section_boundaries])
        utils.writelog("%s: {CoNLL vs. BioNLP: OK}" % (filename.replace(".conll", "")))

        # CoNLLデータの保存
        lines = utils.read_lines(os.path.join(path_conll, filename))
        begin_line = lines[0]
        end_line = "#end document"
        assert begin_line.startswith("#begin document (")
        with open(os.path.join(path_dst + "-conll", split, filename), "w") as f:
            f.write("%s\n" % begin_line)
            for sent in sentences_conll:
                for conll_line in sent:
                    items = [conll_line[key] for key in conll_line.keys()]
                    f.write("\t".join(items) + "\n")
                f.write("\n")
            f.write("%s\n" % end_line)

        # テキストデータに変換、保存
        sections_tok, sections_pos = get_sections(sentences_conll, inner_section_boundaries)
        write_sections(os.path.join(path_dst, split, filename.replace(".conll", ".inner_sections.tokens")), sections_tok)
        write_sections(os.path.join(path_dst, split, filename.replace(".conll", ".inner_sections.postags")), sections_pos)

        sections_tok, sections_pos = get_sections(sentences_conll, outer_section_boundaries)
        write_sections(os.path.join(path_dst, split, filename.replace(".conll", ".outer_sections.tokens")), sections_tok)
        write_sections(os.path.join(path_dst, split, filename.replace(".conll", ".outer_sections.postags")), sections_pos)

        # セクション境界、文境界の保存
        write_boundaries(os.path.join(path_dst, split, filename.replace(".conll", ".inner_section_boundaries")), inner_section_boundaries)
        write_boundaries(os.path.join(path_dst, split, filename.replace(".conll", ".outer_section_boundaries")), outer_section_boundaries)
        sentence_boundaries = get_sentence_boundaries(sentences_conll)
        write_boundaries(os.path.join(path_dst, split, filename.replace(".conll", ".sentence_boundaries")), sentence_boundaries)

        # Chains抽出
        tokens = [w["token"] for s in sentences_conll for w in s]
        sentences_conll = assign_global_token_index(sentences_conll)
        chains = extract_identity_chains(sentences_conll)
        chains = assign_text_to_mensions(chains, tokens)

        # 辞書に変換、保存
        data = get_chains_dictionary(chains)
        utils.write_json(os.path.join(path_dst, split, filename.replace(".conll", ".chains.json")), data)

        # カウント
        n_sents += len(sentences_conll)
        n_chains += len(chains)
        for chain in chains:
            n_mensions += len(chain)

    utils.writelog("No. sentences = %d" % n_sents)
    utils.writelog("Avg. sentences/doc = %f" % (float(n_sents) / n_doc))
    utils.writelog("No. mensions = %d" % n_mensions)
    utils.writelog("Avg. mensions/doc = %f" % (float(n_mensions) / n_doc))
    utils.writelog("No. chains = %d" % n_chains)
    utils.writelog("Avg. chains/doc = %f" % (float(n_chains) / n_doc))

#########################
def read_section_boundaries_from_bionlp(path):
    outer_section_boundaries = []
    inner_section_boundaries = []

    lines = utils.read_lines(path, process=lambda line: line.split("\t"))
    lines = [l for l in lines if l[0].startswith("T")]

    section_charlevel_boundaries = get_charlevel_boundaries(lines, target_tag="section")
    # outer/innerのセクションを削除
    inner_section_charlevel_boundaries = remove_outer_boundaries(section_charlevel_boundaries)
    outer_section_charlevel_boundaries = remove_inner_boundaries(section_charlevel_boundaries)
    # セクション間を補間する
    inner_section_charlevel_boundaries = fill_boundaries(inner_section_charlevel_boundaries)
    outer_section_charlevel_boundaries = fill_boundaries(outer_section_charlevel_boundaries)

    sentence_charlevel_boundaries = get_charlevel_boundaries(lines, target_tag="sentence")

    # 各セクションが含む文単位スパンを同定
    for sec_i in range(len(inner_section_charlevel_boundaries)):
        sec_begin_char_i, sec_end_char_i = inner_section_charlevel_boundaries[sec_i]
        sent_indices = []
        for sent_i in range(len(sentence_charlevel_boundaries)):
            sent_begin_char_i, sent_end_char_i = sentence_charlevel_boundaries[sent_i]
            if sec_begin_char_i <= sent_begin_char_i < sent_end_char_i <= sec_end_char_i:
                sent_indices.append(sent_i)
        if len(sent_indices) != 0:
            inner_section_boundaries.append((min(sent_indices), max(sent_indices)))

    for sec_i in range(len(outer_section_charlevel_boundaries)):
        sec_begin_char_i, sec_end_char_i = outer_section_charlevel_boundaries[sec_i]
        sent_indices = []
        for sent_i in range(len(sentence_charlevel_boundaries)):
            sent_begin_char_i, sent_end_char_i = sentence_charlevel_boundaries[sent_i]
            if sec_begin_char_i <= sent_begin_char_i < sent_end_char_i <= sec_end_char_i:
                sent_indices.append(sent_i)
        if len(sent_indices) != 0:
            outer_section_boundaries.append((min(sent_indices), max(sent_indices)))

    return inner_section_boundaries, outer_section_boundaries

def get_charlevel_boundaries(lines, target_tag):
    boundaries = []
    for line in lines:
        assert len(line) == 3
        typ, tag_and_boundary, _  = line

        if not typ.startswith("T"):
            continue
        if not tag_and_boundary.startswith(target_tag):
            continue

        elements = tag_and_boundary.split(" ")
        assert len(elements) == 3
        tag, begin_char_i, end_char_i = elements
        begin_char_i = int(begin_char_i)
        end_char_i = int(end_char_i)

        boundaries.append((begin_char_i, end_char_i))
    return boundaries

def remove_outer_boundaries(boundaries):
    remove_indices = []
    for i in range(len(boundaries)):
        i_begin, i_end = boundaries[i]
        for j in range(len(boundaries)):
            if i == j:
                continue
            j_begin, j_end = boundaries[j]
            # もしi番目のboundaryがj番目のboundaryを含んでいれば, "i"番目を削除リストに入れる
            if i_begin <= j_begin < j_end <= i_end:
                remove_indices.append(i)
    new_boundaries = []
    for i in range(len(boundaries)):
        if not i in remove_indices:
            new_boundaries.append(boundaries[i])
    return new_boundaries

def remove_inner_boundaries(boundaries):
    remove_indices = []
    for i in range(len(boundaries)):
        i_begin, i_end = boundaries[i]
        for j in range(len(boundaries)):
            if i == j:
                continue
            j_begin, j_end = boundaries[j]
            # もしi番目のboundaryがj番目のboundaryを含んでいれば, "j"番目を削除リストに入れる
            if i_begin <= j_begin < j_end <= i_end:
                remove_indices.append(j)
    new_boundaries = []
    for i in range(len(boundaries)):
        if not i in remove_indices:
            new_boundaries.append(boundaries[i])
    return new_boundaries

def fill_boundaries(boundaries):
    new_boundaries = []
    for i in range(len(boundaries)-1):
        cur_begin, cur_end = boundaries[i]
        next_begin, next_end = boundaries[i+1]
        # セクションが連続していなければ、隙間を埋める
        assert cur_end <= next_begin
        if next_begin != cur_end:
            new_boundaries.append((cur_end, next_begin))
    boundaries = boundaries + new_boundaries
    boundaries = sorted(boundaries, key=lambda  x: x[0])
    return boundaries

#########################
def get_sections(sentences_conll, section_boundaries):
    sections_tok = []
    sections_pos = []
    for begin_i, end_i in section_boundaries:
        section_tok = []
        section_pos = []
        for sent_conll in sentences_conll[begin_i:end_i+1]:
            tokens = [w["token"] for w in sent_conll]
            postags = [w["pos"] for w in sent_conll]
            section_tok.append(tokens)
            section_pos.append(postags)
        sections_tok.append(section_tok)
        sections_pos.append(section_pos)
    return sections_tok, sections_pos

def write_sections(path, sections):
    with open(path, "w") as f:
        for section in sections:
            # 1セクションの書き込み; 各行は文
            for sent in section:
                sent = " ".join(sent)
                f.write("%s\n" % sent)
            # セクションは空行で区切られる
            f.write("\n")

def write_tokens(path, tokens):
    with open(path, "w") as f:
        for token in tokens:
            f.write("%s\n" % token)

#########################
def get_sentence_boundaries(sentences_conll):
    sentence_boundaries = []
    begin_i = end_i = 0
    for sent_conll in sentences_conll:
        n_tokens = len(sent_conll)
        end_i = begin_i + n_tokens - 1
        sentence_boundaries.append((begin_i, end_i))
        begin_i = end_i + 1
    return sentence_boundaries

def write_boundaries(path, boundaries):
    with open(path, "w") as f:
        for begin_i, end_i in boundaries:
            f.write("%d %d\n" % (begin_i, end_i))

#########################
class Mension(object):

    def __init__(self, chain_name, mension_name, spans, text=None):
        self.chain_name = chain_name
        self.mension_name = mension_name
        self.spans = spans
        self.text = text

    def __str__(self):
        text = "<%s, spans=%s, chain_name=%s, mension_name=%s>" % (self.text, self.spans, self.chain_name, self.mension_name)
        return text

    def __repr__(self):
        return self.__str__()

class ChainManager(object):

    def __init__(self):
        self.stacks = defaultdict(list)
        self.chains = defaultdict(list)

    def get_key(self, chain_name, mension_name):
        return "%s-%s" % (chain_name, mension_name)

    def set_begin_index(self, chain_name, mension_name, begin_index):
        key = self.get_key(chain_name, mension_name)
        self.stacks[key].append((begin_index, "*"))

    def set_end_index(self, chain_name, mension_name, end_index):
        key = self.get_key(chain_name, mension_name)
        assert key in self.stacks
        assert len(self.stacks[key]) > 0
        begin_index, _ = self.stacks[key].pop()
        self.chains[key].append((begin_index, end_index))

        # To avoid errors in get_chains(), we initialize the element for (chain_name=any, mension_name="")
        key2 = self.get_key(chain_name, "")
        if not key2 in self.chains:
            self.chains[key2] = []

    def get_chains(self):
        # chains = [self.chains[chain_name] for chain_name in self.chains]
        # return chains
        new_chains = {}
        for key in self.chains:
            chain_name, mension_name = key.split("-")
            if not chain_name in new_chains:
                new_chains[chain_name] = []
            if mension_name == "":
                for span in self.chains[key]:
                    mension = Mension(chain_name=chain_name, mension_name=mension_name, spans=[span], text=None)
                    new_chains[chain_name].append(mension)
            else:
                discont_spans = self.chains[key]
                mension = Mension(chain_name=chain_name, mension_name=mension_name, spans=discont_spans, text=None)
                new_chains[chain_name].append(mension)
        return list(new_chains.values())

def assign_global_token_index(sentences_conll):
    """
    Assign global token indices to token-level dictionaries extracted from the CoNLL-format file
    """
    token_index = 0
    for s_i in range(len(sentences_conll)):
        for w_i in range(len(sentences_conll[s_i])):
            sentences_conll[s_i][w_i]["global-token-index"] = token_index
            token_index += 1
    return sentences_conll

def extract_identity_chains(sentences_conll):
    """
    Extract identity chains (ie., a set of mention spans) from a CoNLL-format data
    """
    manager = ChainManager()
    for sent_conll in sentences_conll:
        for word_conll in sent_conll:
            span_exps = word_conll["span-exp"]
            global_token_index = word_conll["global-token-index"]
            span_exps = span_exps.split("|")
            for span_exp in span_exps:
                if span_exp.startswith("(") and span_exp.endswith(")"):
                    chain_name = re.findall(r"\(([0-9]+)([a-z]*)\)", span_exp)
                    assert len(chain_name) == 1
                    chain_name, mension_name = chain_name[0]
                    manager.set_begin_index(chain_name=chain_name, mension_name=mension_name, begin_index=global_token_index)
                    manager.set_end_index(chain_name=chain_name, mension_name=mension_name, end_index=global_token_index)
                elif span_exp.startswith("("):
                    chain_name = re.findall(r"\(([0-9]+)([a-z]*)", span_exp)
                    assert len(chain_name) == 1
                    chain_name, mension_name = chain_name[0]
                    manager.set_begin_index(chain_name=chain_name, mension_name=mension_name, begin_index=global_token_index)
                elif span_exp.endswith(")"):
                    chain_name = re.findall(r"([0-9]+)([a-z]*)\)", span_exp)
                    assert len(chain_name) == 1
                    chain_name, mension_name = chain_name[0]
                    manager.set_end_index(chain_name=chain_name, mension_name=mension_name, end_index=global_token_index)
    chains = manager.get_chains()
    return chains

def assign_text_to_mensions(chains, tokens):
    for c_i in range(len(chains)):
        for m_i in range(len(chains[c_i])):
            chains[c_i][m_i].text = " ".join([" ".join(tokens[i:j+1]) for (i,j) in chains[c_i][m_i].spans])
    return chains

#########################
def get_chains_dictionary(chains):
    data = {}
    data["chains"] = []
    for chain in chains:
        chain = [["%d %d" % (b,e) for b,e in m.spans] for m in chain]
        data["chains"].append(chain)
    data["chains_text"] = []
    for chain in chains:
        chain = [m.text for m in chain]
        data["chains_text"].append(chain)
    return data

if __name__ == "__main__":
    main()
