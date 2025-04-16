#!/bin/bash

python3 ./data/data_labeler.py --folder "kinship_family" --use_tabling
#python data_labeler.py --folder "countries" --level "s2" --use_tabling

# s1 does not need tabling
#python data_labeler.py --folder "countries" --level "s1"