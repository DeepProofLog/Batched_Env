#!/bin/bash

python data_labeler.py --task "all" --folder "ablation" --level "d3" --collapse_type "tail" --sample_ratio "[[1, 0], [1, 0], [1, 0]]"
python data_labeler.py --task "all" --folder "countries" --level "s3" --collapse_type "tail" --sample_ratio "[[1, 0], [1, 0], [1, 0]]"