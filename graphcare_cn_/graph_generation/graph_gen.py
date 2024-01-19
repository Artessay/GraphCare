import os
import re
import csv
from tqdm import tqdm
from ChatGPT import ChatGPT

# %% [markdown]
# 分别加载电子病历内condition、procedure和drug中CCS-CM、CCS-PROC和ATC编码，构建字典以供后续使用。

# %%
condition_mapping_file = "../../resources/CCSCM.csv"
procedure_mapping_file = "../../resources/CCSPROC.csv"
drug_file = "../../resources/ATC.csv"

condition_dict = {}
with open(condition_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        condition_dict[row['code']] = row['name'].lower()

procedure_dict = {}
with open(procedure_mapping_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        procedure_dict[row['code']] = row['name'].lower()

drug_dict = {}
with open(drug_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['level'] == '3.0':
            drug_dict[row['code']] = row['name'].lower()


# %% [markdown]
# 询问LLM在一系列三元组中最重要的三元组是哪些，选出不超过50个三元组

# %%
# 抽取在[]中的所有字符串
def extract_data_in_brackets(input_string):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches

# 将长文本切分成以max_len为最大长度的多个片段
def divide_text(long_text, max_len=800):
    sub_texts = []
    start_idx = 0
    while start_idx < len(long_text):
        end_idx = start_idx + max_len
        sub_text = long_text[start_idx:end_idx]
        sub_texts.append(sub_text)
        start_idx = end_idx
    return sub_texts

# 过滤三元组
def filter_triples(triples):
    chatgpt = ChatGPT()
    response = chatgpt.chat(
        f"""
            I have a list of triples. I want to select 50 most important triples from the list.
            The importance of a triple is based on how you think it will help imrpove healthcare prediction tasks (e.g., drug recommendation, mortality prediction, readmission prediction …).
            If you think a triple is important, please keep it. Otherwise, please remove it.
            You can also add triples from your background knowledge.
            The total size of the updated list should be below 50.

            triples: {triples}
            updates:
        """
        )

    filtered_triples = extract_data_in_brackets(response)
    return filtered_triples


# %% [markdown]
# 从给定的term中让LLM推演出相关的三元组

# %%
def graph_gen(term: str, mode: str):
    if mode == "condition":
        example = \
        """
        Example:
        prompt: systemic lupus erythematosus
        updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for, systemic lupus erythematosus], [systemic lupus erythematosus, is treated with, steroids], [methylprednisolone, is a, steroid]]
        """
    elif mode == "procedure":
        example = \
        """
        Example:
        prompt: endoscopy
        updates: [[endoscopy, is a, medical procedure], [endoscopy, used for, diagnosis], [endoscopic biopsy, is a type of, endoscopy], [endoscopic biopsy, can detect, ulcers]]
        """
    elif mode == "drug":
        example = \
        """
        Example:
        prompt: iobenzamic acid
        updates: [[iobenzamic acid, is a, drug], [iobenzamic acid, may have, side effects], [side effects, can include, nausea], [iobenzamic acid, used as, X-ray contrast agent], [iobenzamic acid, formula, C16H13I3N2O3]]
        """
    chatgpt = ChatGPT()
    response = chatgpt.chat(
        f"""
            Given a prompt (a medical condition/procedure/drug), extrapolate as many relationships as possible of it and provide a list of updates.
            The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …)
            Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
            Both ENTITY 1 and ENTITY 2 should be noun.
            Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as short as possible.
            Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until the size reaches 100.

            {example}

            prompt: {term}
            updates:
        """
        )

    triples = extract_data_in_brackets(response)
    outstr = ""
    for triple in triples:
        outstr += triple.replace('[', '').replace(']', '').replace(', ', '\t') + '\n'

    return outstr

# %% [markdown]
# 为每一个condition建图

# %%
dir_path = '../../graphs/condition/CCSCM/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for key in tqdm(condition_dict.keys()):
    file = f'../../graphs/condition/CCSCM/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 100:
            outstr = graph_gen(term=condition_dict[key], mode="condition")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
    else:
        outstr = graph_gen(term=condition_dict[key], mode="condition")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)

# %%
dir_path = '../../graphs/procedure/CCSPROC/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for key in tqdm(procedure_dict.keys()):
    file = f'../../graphs/procedure/CCSPROC/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 150:
            outstr = graph_gen(term=procedure_dict[key], mode="procedure")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
    else:
        outstr = graph_gen(term=procedure_dict[key], mode="procedure")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)

# %%
dir_path = '../../graphs/drug/ATC5/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for key in tqdm(drug_dict.keys()):
    file = f'../../graphs/drug/ATC5/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 150:
            outstr = graph_gen(term=drug_dict[key], mode="drug")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
        # continue
    else:
        outstr = graph_gen(term=drug_dict[key], mode="drug")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)

# %%
for key in tqdm(drug_dict.keys()):
    file = f'../../graphs/drug/ATC3/{key}.txt'
    if os.path.exists(file):
        with open(file=file, mode="r", encoding='utf-8') as f:
            prev_triples = f.read()
        if len(prev_triples.split('\n')) < 150:
            outstr = graph_gen(term=drug_dict[key], mode="drug")
            outfile = open(file=file, mode='w', encoding='utf-8')
            outstr = prev_triples + outstr
            # print(outstr)
            outfile.write(outstr)
        # continue
    else:
        outstr = graph_gen(term=drug_dict[key], mode="drug")
        outfile = open(file=file, mode='w', encoding='utf-8')
        outstr = outstr
        # print(outstr)
        outfile.write(outstr)
