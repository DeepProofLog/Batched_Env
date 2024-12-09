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
   - countries.pl: consulted by janus
   - train_label.txt, test_label.txt, valid_label.txt
  
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
   - train_queries.txt, test_queries.txt, valid_queries.txt
