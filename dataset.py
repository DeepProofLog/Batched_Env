from os.path import join 
import janus_swi as janus
from typing import List,Tuple,Dict
import re
from utils import is_variable,Term
import json
from collections import defaultdict
import re

import math
import torch
from utils import is_variable,Term
class Rule:
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __str__(self):
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"    

    def __repr__(self):
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}"    
    

def get_atom_from_string(atom_str: str) -> Term:
    predicate, args = atom_str.split("(")
    args = args[:-1].split(",")
    # remove any  ")" in the strings in args
    args = [re.sub(r'\)', '', arg) for arg in args]
    return Term(predicate, args)

def get_max_arity(file_path:str)-> int:
    '''Get the maximum arity of the predicates in the file'''
    max_arity = 0
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            clauses = re.findall(r'\w+\(.*?\)', line)
            for clause in clauses:
                predicate, args = clause.split("(")
                arity = len(args.split(","))
                if arity > max_arity:
                    max_arity = arity
    return max_arity


def get_rules_from_file(file_path: str) -> Tuple[List[Term], List[Rule]]:
    """Get rules from a file"""
    rules = []
    queries = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # if there's no :-, it's a fact, split predicate
            if ":-" not in line:
                head = line.strip()
                queries.append(get_atom_from_string(head))
            elif line.startswith(":-"):
                continue
            else:
                head, body = line.strip().split(":-")
                body = re.findall(r'\w+\(.*?\)', body)
                body = [get_atom_from_string(b) for b in body]
                head_atom = get_atom_from_string(head)
                rules.append(Rule(head_atom, body))
    rules = [rule for rule in rules if not rule.head.predicate != "proof_first"]
    return queries,rules



def get_queries_labels(path:str)-> Tuple[List[Term],List[bool]]:
    '''Get queries and labels (whether they are provable) from a file'''
    queries = []
    labels = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            query, label = line.strip().split("\t")
            queries.append(get_atom_from_string(query))
            labels.append(1 if label == "True" else (0 if label == "False" else '?'))
    return queries, labels

def get_queries(path:str)-> Tuple[List[Term],List[Term]]:
    '''Get queries from a file, and the corresponding negative queries (togehter with their labels indicating if they are provable)'''
    pos_queries = []
    neg_queries = []
    with open(path, "r") as f:
        dicts = json.load(f)
        for key, value in dicts.items():
            pos_queries.append(get_atom_from_string(key))
            for q, l in value:
                if l:
                    neg_queries.append(get_atom_from_string(q))
    return pos_queries, neg_queries

def get_predicates_and_constants(rules: List[Rule], facts: List[Term]) -> Tuple[set[str], set[str], dict[str,int]]:
    predicates = set()
    constants = set()
    predicates_arity = {}
    for rule in rules:
        # proof_first not related to query generation
        if not rule.head.predicate == "proof_first":
            # predicates.add((rule.head.predicate, len(rule.head.args)))
            predicate = rule.head.predicate
            predicates.add((predicate))
            constants.update([arg for arg in rule.head.args if not is_variable(arg)])
            if predicate not in predicates_arity:
                predicates_arity[predicate] = len(rule.head.args)
            for atom in rule.body:
                predicates.add((atom.predicate))
                constants.update([arg for arg in atom.args if not is_variable(arg)])
                if atom.predicate not in predicates_arity:
                    predicates_arity[atom.predicate] = len(atom.args)
    for atom in facts:
        # predicates.add((atom.predicate, len(atom.args)))
        predicates.add((atom.predicate))
        constants.update([arg for arg in atom.args if not is_variable(arg)])
        if atom.predicate not in predicates_arity:
            predicates_arity[atom.predicate] = len(atom.args)
    return predicates, constants, predicates_arity

def get_corruptions(queries: List[Term], file_path: str) -> dict[Term, List[Term]]:
    '''Get corruptions from the json file, in the format
    {"locatedInCR(armenia,asia).": [["locatedInCR(armenia,oceania).", true], ["locatedInCR(armenia,europe).", true], ["locatedInCR(armenia,africa).", true], [
    return a dictionary with the query as key, and a list of corruptions that are true as value
    '''
    dict_ = defaultdict(list)
    with open(file_path, "r") as f:
        corruptions_dict = json.load(f)
        for query, corruptions in corruptions_dict.items():
            query = get_atom_from_string(query[:-1])
            assert query in queries, f"Query {query} not in queries"
            for corruption, is_provable in corruptions:
                if is_provable:
                    corruption = get_atom_from_string(corruption[:-1])
                    dict_[query].append(corruption)
    return dict_



class DataHandler():
    '''
    Instead of the normal test,valid txt files, we load a file with the corruptions
    Each query will have a label indicating true or false, i.e. if the query is a corruption or not
    For each set, we will have queries and labels, instead of just queries
    For train, we will load facts from the original train file, and train queries in the same way as valid, test

    if I want to do the original config, with no corruptions, just apply get_rules_from_file to the val,test file
    '''
    def __init__(self, dataset_name: str,
                    base_path: str,
                    janus_file: str,
                    train_file: str = None,
                    valid_file: str = None,
                    test_file: str = None,
                    standard_corruptions: bool = False,

                    dynamic_neg: bool = False,
                    train_neg_pos_ratio: int = 1,
                    name: str = None):

        self.name = name
        self.standard_corruptions = standard_corruptions
        
        base_path  = join(base_path, dataset_name)
        janus_path = join(base_path, janus_file)
        self.janus_path = janus_path

        train_path = join(base_path, train_file) 
        valid_path = join(base_path, valid_file) if valid_file else None
        test_path = join(base_path, test_file)
        janus.consult(janus_path)

        self.facts, self.rules = get_rules_from_file(janus_path)
        if not dynamic_neg:
            self.train_queries, self.train_labels = get_queries_labels(train_path)
        else:
            self.pos_train_queries, self.neg_train_queries = get_queries(train_path)
        self.valid_queries, self.valid_labels = get_queries_labels(valid_path)
        self.test_queries, self.test_labels = get_queries_labels(test_path)

        if not dynamic_neg:
            print('ratio of positives in train', '{:.2f}'.format(sum(self.train_labels)/len(self.train_labels)))
        else:
            print('ratio of positives in train', '{:.2f}'.format(1/(int(train_neg_pos_ratio)+1)))
        print('                      valid', '{:.2f}'.format(sum(self.valid_labels)/len(self.valid_labels)))
        print('                      test', '{:.2f}'.format(sum(self.test_labels)/len(self.test_labels)))

        if dynamic_neg: # dont we already have the train corruptions in neg_train_queries?
            self.train_corruptions = get_corruptions(self.pos_train_queries, join(base_path, "train_label_corruptions.json"))
            self.valid_corruptions = get_corruptions(self.valid_queries, join(base_path, "valid_label_corruptions.json"))
            self.test_corruptions = get_corruptions(self.test_queries, join(base_path, "test_label_corruptions.json"))
        self.janus_facts = []
        with open(janus_path, "r") as f:
            self.janus_facts = f.readlines()
            # lines = f.readlines()
            # print("lines",lines)
            # for line in lines:
                # self.janus_facts.append(line.strip())

        self.predicates, self.constants, self.predicates_arity = get_predicates_and_constants(self.rules, self.facts)
        self.max_arity = get_max_arity(janus_path)
        self.constant_no, self.predicate_no = len(self.constants), len(self.predicates)


        self.entity2domain = None
        self.domain2entity = None
        if standard_corruptions:
            if 'countries' in self.name or 'ablation' in self.name:
                # load the domain file
                domain_file = join(base_path, "domain2constants.txt")
                self.entity2domain = {}
                self.domain2entity = defaultdict(list)
                with open(domain_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        domain, *entities = line.split()
                        for entity in entities:
                            self.entity2domain[entity] = domain
                            self.domain2entity[domain].append(entity)

           





from pykeen.sampling import BasicNegativeSampler
from typing_extensions import TypeAlias 
LongTensor: TypeAlias = torch.LongTensor  

class BasicNegativeSamplerDomain(BasicNegativeSampler):

    def __init__(self, 
                mapped_triples,
                domain2idx: Dict[str, int],
                entity2domain: Dict[int, str], 
                num_negs_per_pos: int = 5,
                filtered: bool = True,
                corruption_scheme: List[str] = ['tail'],
                ):
        """
        Initialize the BasicNegativeSamplerDomain.

        Args:
            mapped_triples: The mapped triples.
            domain2idx (Dict[str, int]): A dictionary mapping domains to indices.
            entity2domain (Dict[int, str]): A dictionary mapping entities to domains.
            num_negs_per_pos (int): The number of negative samples per positive triple.
            filtered (bool): Whether to use filtered negative sampling.
            corruption_scheme (List[str]): The corruption scheme.
        """
        super().__init__(mapped_triples=mapped_triples,
                         num_negs_per_pos=num_negs_per_pos,
                         filtered=filtered,
                         corruption_scheme=corruption_scheme)
        
        self.domain2idx = domain2idx
        self.entity2domain = entity2domain
        self.idx2domain = {idx: domain for domain, idxs in domain2idx.items() for idx in idxs}
        self.domain_entities = {}
        for entity, domain in self.entity2domain.items():
            if domain not in self.domain_entities:
                self.domain_entities[domain] = []
            self.domain_entities[domain].append(entity)
        for domain in self.domain_entities:
            self.domain_entities[domain] = torch.tensor(self.domain_entities[domain], dtype=torch.long, device=mapped_triples.device)


    def corrupt_batch(self, positive_batch: LongTensor) -> LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]
        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)
        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))
        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)

            for i in range(start, stop):
                original_entity = negative_batch[i, index].item()
                original_domain = self.entity2domain[original_entity]

                possible_entities = self.domain_entities[original_domain]
                
                replacement_index = torch.randint(high=len(possible_entities), size=(1,), device=negative_batch.device).item()
                replacement_entity = possible_entities[replacement_index].item()
                while replacement_entity==original_entity: #make sure that the entity is different
                  replacement_index = torch.randint(high=len(possible_entities), size=(1,), device=negative_batch.device).item()
                  replacement_entity = possible_entities[replacement_index].item()
                negative_batch[i, index] = replacement_entity
        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
    

def get_sampler(data_handler: DataHandler, 
                index_manager, 
                triples_factory
                ):

    if 'countries' or 'ablation' in data_handler.dataset_name:
        domain2idx = {domain: [index_manager.constant_str2idx[e] for e in entities] for domain, entities in data_handler.domain2entity.items()}
        entity2domain: Dict[int, str] = {index_manager.constant_str2idx[e]: domain for domain, entities in data_handler.domain2entity.items() for e in entities}
        sampler = BasicNegativeSamplerDomain(mapped_triples=triples_factory.mapped_triples,  # Pass mapped_triples instead
                                            domain2idx=domain2idx,
                                            entity2domain=entity2domain,
                                            num_negs_per_pos=1,
                                            filtered=True,
                                            corruption_scheme=['tail'],)
    else:
        sampler = BasicNegativeSampler(mapped_triples=triples_factory.mapped_triples,  # Pass mapped_triples instead
                                    num_negs_per_pos=1,
                                    filtered=True,
                                    corruption_scheme=['tail'])
    
    return sampler

