import os
import janus_swi as janus
import re
import argparse
from pathlib import Path
import json
import ast
import random
random.seed(42)



def generate_corruptions(folder, level, corruption_type, full_gt):

    root_dir = f"{folder}_{level}/"
    domain_dir = root_dir + "domain2constants.txt"

    with open(domain_dir, "r") as f:
        domains = f.readlines()
        tail_domain = domains[0].strip().split()[1:]
        head_domain = domains[1].strip().split()[1:]

    knowledge_dir = root_dir+"countries.pl"
    #janus.consult(knowledge_dir)
    with open(knowledge_dir, "r") as f:
        knowledge = f.readlines()
        knowledge = [k.strip() for k in knowledge]
    # print(len(knowledge))

    file_list = ["train_label.txt", "test_label.txt", "valid_label.txt"]

    for file in file_list:
        outputs = {}
        with open(root_dir+file, "r") as f:
            queries = f.readlines()
            for q in queries:
                # print(q)
                # sub_knowledge = knowledge.copy()
                query, value = q.strip().split("\t")
                if value == "True":
                    if query not in outputs:
                        outputs[query] = []
                    matches = re.findall(r'(\b\w+)\(([^)]*)\)', query)
                    for m in matches:
                        country, region = m[1].split(",")
                        if corruption_type == "tail":
                            corruptions = [f'locatedInCR({country},{r}).' for r in tail_domain if r != region]
                        elif corruption_type == "head":
                            corruptions = [f'locatedInCR({c},{region}).' for c in head_domain if c != country]
                        corruptions = [c for c in corruptions if c not in full_gt]
                        for c in corruptions:
                            sub_knowledge = knowledge.copy()
                            if c in sub_knowledge:
                                sub_knowledge.remove(c)
                            with open("countries_sub.pl", "w") as f1:
                                # print(len(sub_knowledge))
                                f1.write("\n".join(sub_knowledge))
                            f1.close()
                            janus.query_once("abolish(neighborOf/2).")
                            janus.query_once("abolish(locatedInCR/2).")
                            janus.query_once("abolish_all_tables.")
                            janus.consult("countries_sub.pl")
                            res = janus.query_once(c)
                            # print(f"{c}\t{res['truth']}")
                            outputs[query].append([c, res['truth']])

        of_dir = root_dir+file.split(".")[0] + "_corruptions.json"
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
                # if value == "True":
                full_gt.add(query)

    return full_gt

def prepare_queries(folder, level, sample_type, sample_ratio):

    sample_type = sample_type[1:-1].split(",")
    sample_type = [s.strip() for s in sample_type]
    sample_ratio = ast.literal_eval(sample_ratio)

    root_dir = f"{folder}_{level}/"
    file_list = ["train_label_corruptions.json", "test_label_corruptions.json", "valid_label_corruptions.json"]
    for i in range(len(file_list)):
        with open(root_dir+file_list[i], "r") as f:
            corruptions = json.load(f)
        outputs = []
        all_provable_false = []
        all_non_provable = []
        for query, corruption in corruptions.items():
            outputs.append(f"{query}\tTrue\n")
            provable_false = [c[0] for c in corruption if c[1]]
            non_provable = [c[0] for c in corruption if not c[1]]
            all_provable_false.extend(provable_false)
            all_non_provable.extend(non_provable)
            if sample_type[i] == "paired":
                if int(sample_ratio[i][0]) > len(provable_false):
                    outputs.extend([f"{q}\tFalse\n" for q in provable_false])
                else:
                    outputs.extend([f"{q}\tFalse\n" for q in random.sample(provable_false, int(sample_ratio[i][0]))])
                if int(sample_ratio[i][1]) > len(non_provable):
                    outputs.extend(f"{q}\tNon-provable\n" for q in non_provable)
                else:
                    outputs.extend([f"{q}\tNon-provable\n" for q in random.sample(non_provable, int(sample_ratio[i][1]))])
            if sample_type[i] == "all_possible":
                outputs.extend([f"{q}\tFalse\n" for q in provable_false])
            if sample_type[i] == "all_possible_both":
                outputs.extend([f"{q}\tFalse\n" for q in provable_false])
                outputs.extend(f"{q}\tNon-provable\n" for q in non_provable)
        if sample_type[i] == "full_set":
            provable_true_no = len(outputs)
            #print(len(outputs), len(all_provable_false), len(all_non_provable))
            if int(sample_ratio[i][0])*provable_true_no > len(all_provable_false):
                outputs.extend([f"{q}\tFalse\n" for q in all_provable_false])
            else:
                outputs.extend([f"{q}\tFalse\n" for q in random.sample(all_provable_false, int(sample_ratio[i][0])*provable_true_no)])
            if int(sample_ratio[i][1])*provable_true_no > len(all_non_provable):
                outputs.extend(f"{q}\tNon-provable\n" for q in all_non_provable)
            else:
                outputs.extend([f"{q}\tNon-provable\n" for q in random.sample(all_non_provable, int(sample_ratio[i][1])*provable_true_no)])
        #random.shuffle(outputs)
        of_dir = root_dir+file_list[i].split("_")[0] + "_queries.txt"
        with open(of_dir, "w") as of:
            of.writelines(outputs)





if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', default='all', type=str, help="generate_corruptions, prepare_queries, all")
    arg_parser.add_argument('--folder', default='ablation', type=str)
    arg_parser.add_argument('--level', default='d3', type=str)
    arg_parser.add_argument('--corruption_type', default='tail', type=str)
    arg_parser.add_argument('--sample_type', default='[full_set, all_possible, all_possible]', type=str, help="full_set, paired, all_possible, all_possible_both")
    arg_parser.add_argument('--sample_ratio', default='[[1, 0], [1, 0], [1, 0]]', type=str, help="ratio of provable false / provable true and unprovable / provable true for train, test, valid")
    args = arg_parser.parse_args()

    if args.task == "all" or args.task == "generate_corruptions":
        full_gt = get_full_gt()
        generate_corruptions(args.folder, args.level, args.corruption_type, full_gt)

    if args.task == "all" or args.task == "prepare_queries":
        prepare_queries(args.folder, args.level, args.sample_type, args.sample_ratio)



