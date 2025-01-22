from environments.env_logic_gym import IndexManager
from dataset import DataHandler, BasicNegativeSamplerDomain
from utils import get_device
import numpy as np
import random
import torch

# In runner.py do: from corruptions import main 

def main(args,log_filename,use_logger,use_WB,WB_path,date):

    torch.manual_seed(args.seed_run_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_run_i)
    random.seed(args.seed_run_i)
    np.random.seed(args.seed_run_i)

    device = get_device(args.device)

    data_handler = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        use_only_positives=args.only_positives,
        use_validation_as_train=False,
        dynamic_neg=args.dynamic_neg,
        train_neg_pos_ratio=args.train_neg_pos_ratio,
        name=args.dataset_name)

    index_manager = IndexManager(data_handler.constants, 
                                 data_handler.predicates,
                                 data_handler.constant_no, 
                                 data_handler.predicate_no,
                                 args.variable_no,
                                 max_arity=data_handler.max_arity, 
                                 device=device)





    from pykeen.sampling import BasicNegativeSampler
    from pykeen.triples import TriplesFactory
    # from corruptions import BasicNegativeSamplerDomain
    np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.facts],dtype=str)
    triples_factory = TriplesFactory.from_labeled_triples(triples=np_facts,
                                                        entity_to_id=index_manager.constant_str2idx,
                                                        relation_to_id=index_manager.predicate_str2idx,
                                                        compact_id=False,
                                                        create_inverse_triples=False)

    from dataset import get_sampler

    sampler = get_sampler(data_handler, index_manager, triples_factory=triples_factory)
    
    data_handler.sampler = sampler

    query = [('taiwan', 'locatedInCR', 'asia'), ('spain', 'locatedInCR', 'europe')]

    positive_batch = triples_factory.map_triples(np.array(query))
    negative_batch = sampler.corrupt_batch(positive_batch)

    print('positive batch:',positive_batch.shape,positive_batch)
    print('negative batch:',negative_batch.shape,negative_batch)

    negative_batch_str = []
    for batch in negative_batch:
        for n in batch:
            negative_batch_str.append((index_manager.constant_idx2str[n[0].item()],index_manager.predicate_idx2str[n[1].item()],
                                    index_manager.constant_idx2str[n[2].item()]))
            
    print('positive batch:',query)
    print('negative batch:',negative_batch_str)

    print('Ending test_corruptions.py',end)