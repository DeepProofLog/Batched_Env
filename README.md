# Neural-guided-Grounding

## Data preparation

### Label queries in train, test, valid separation of original dataset, test whether they are provable or not with Prolog
```
bash data_labeler.sh
```
1. Arguments:
   - folder: e.g. "ablation"
   - level: e.g. "d3"
   - use_tabling: include this when there are recursive predicates, avoid inifite loops
   - catch errors: include this to catch time out and stack overflow errors from Prolog side, not used now
   - use_modified_rules: include this to put recursive predicates on the right-most, not used now

2. Outputs:
   - countries.pl: created by data_labeller and used in data_preparation
   - train_label.txt, test_label.txt, valid_label.txt
      - Composed of the same queries as original file, with label True (they are provable) and False (they are not provable)
  
### Prepare train, test, valid queries. Include two tasks: generate corruptions and sample corruptions to build query files
```
bash data_preparation.sh
```
1. Arguments:
   - task: "all" / "generate_corruptions" / "prepare_queries"
   - folder: e.g. "ablation"
   - level: e.g. "d3"
   - corruption_type: "tail" / "head"
   - sample_type: "full_set" - sample according to ratio on set level / "paired" - sample according to ratio for each provable true query / "all_possible" - include all provable false / "all_possible_both" - include all provable false and non-provable, e.g. '[full_set, all_possible, all_possible]'
   - sample_ration: ratio of provable false / provable true and non-provable / provable true for train, test, valid, e.g. "[[1, 0], [1, 0], [1, 0]]"

3. Outputs:
   - train_label_corruptions.json, test_label_corruptions.json, valid_label_corruptions.json
      - For each query from train/val/test, it has all its corruptions, and whether they are provable or not
   - train_queries.txt, test_queries.txt, valid_queries.txt
      - Composed of original queries from train/val/test with corruptions that can be provable or not 
      - Labels: True (provable true, i.e. query from dataset), False (provable False, i.e. corruptions provable), non-provable-False (non-provable False, i.e. corruptions non provable)
