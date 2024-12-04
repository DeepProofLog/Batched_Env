#!/bin/bash

python data_labeler.py --folder "ablation" --level "d3" --use_tabling
#python data_labeler.py --folder "countries" --level "s2" --use_tabling

# s1 does not need tabling
#python data_labeler.py --folder "countries" --level "s1"