import os
import janus_swi as janus
import re
import argparse
from pathlib import Path
import json
import ast
import random
random.seed(42)



def generate_collapses(folder, level, collapse_type, full_gt):

    root_dir = f"{folder}_{level}/"
    domain_dir = root_dir + "domain2constants.txt"

    with open(domain_dir, "r") as f:
        domains = f.readlines()
        tail_domain = domains[0].strip().split()[1:]
        head_domain = domains[1].strip().split()[1:]

    knowledge_dir = root_dir+"countries.pl"
    janus.consult(knowledge_dir)

    file_list = ["train_label.txt", "test_label.txt", "valid_label.txt"]

    for file in file_list:
        outputs = {}
        with open(root_dir+file, "r") as f:
            queries = f.readlines()
            for q in queries:
                query, value = q.strip().split("\t")
                if query not in outputs:
                    outputs[query] = []
                if value == "True":
                    matches = re.findall(r'(\b\w+)\(([^)]*)\)', query)
                    for m in matches:
                        country, region = m[1].split(",")
                        if collapse_type == "tail":
                            collapses = [f'locatedInCR({country},{r}).' for r in tail_domain if r != region]
                        elif collapse_type == "head":
                            collapses = [f'locatedInCR({c},{region}).' for c in head_domain if c != country]
                        collapses = [c for c in collapses if c not in full_gt]
                        for c in collapses:
                            res = janus.query_once(c)
                            #print(f"{c}\t{res['truth']}")
                            outputs[query].append([c, res['truth']])

        of_dir = root_dir+file.split(".")[0] + "_collapses.json"
        with open(of_dir, "w") as of:
            json.dump(outputs, of)



def get_full_gt():
    full_gt = set()
    current_dir = Path(".")
    label_files = list(current_dir.rglob("*label*.txt"))

    for file_path in label_files:
        with open(file_path, "r") as file:
            for line in file.readlines():
                query, value = line.strip().split("\t")
                if value == "True":
                    full_gt.add(query)

    return full_gt

def prepare_queries(folder, level, sample_ratio):

    sample_ratio = ast.literal_eval(sample_ratio)

    root_dir = f"{folder}_{level}/"
    file_list = ["train_label_collapses.json", "test_label_collapses.json", "valid_label_collapses.json"]
    for i in range(len(file_list)):
        with open(root_dir+file_list[i], "r") as f:
            collapses = json.load(f)
        outputs = []
        for query, collapse in collapses.items():
            outputs.append(f"{query}\tTrue\n")
            provable_false = [c[0] for c in collapse if c[1]]
            non_provable = [c[0] for c in collapse if not c[1]]
            if int(sample_ratio[i][0]) > len(provable_false):
                outputs.extend(provable_false)
            else:
                outputs.extend([f"{q}\tFalse\n" for q in random.sample(provable_false, int(sample_ratio[i][0]))])
            if int(sample_ratio[i][1]) > len(non_provable):
                outputs.extend(non_provable)
            else:
                outputs.extend([f"{q}\tNon-provable\n" for q in random.sample(non_provable, int(sample_ratio[i][1]))])
        of_dir = root_dir+file_list[i].split("_")[0] + "_queries.txt"
        with open(of_dir, "w") as of:
            of.writelines(outputs)





if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', default='all', type=str)
    arg_parser.add_argument('--folder', default='countries', type=str)
    arg_parser.add_argument('--level', default='s3', type=str)
    arg_parser.add_argument('--collapse_type', default='tail', type=str)
    arg_parser.add_argument('--sample_ratio', default='[[1, 0], [1, 0], [1, 0]]', type=str, help="ratio of provable false / provable true and unprovable / provable true for train, test, valid")
    args = arg_parser.parse_args()

    if args.task == "all" or args.task == "generate_collapses":
        full_gt = get_full_gt()
        generate_collapses(args.folder, args.level, args.collapse_type, full_gt)

    if args.task == "all" or args.task == "prepare_queries":
        prepare_queries(args.folder, args.level, args.sample_ratio)



