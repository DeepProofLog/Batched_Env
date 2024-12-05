#!/bin/bash

python data_labeler.py --task "all" --folder "ablation" --level "d3" --corruption_type "tail" --sample_type "full_set" --sample_ratio "[[1, 0], [1, 0], [1, 0]]"
#python data_labeler.py --task "all" --folder "countries" --level "s3" --corruption_type "tail" --sample_type "full_set" --sample_ratio "[[1, 0], [1, 0], [1, 0]]"