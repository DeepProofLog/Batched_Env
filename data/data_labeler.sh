#!/bin/bash

# s2 & s3 have queries lead to timeout and stack exceed errors
# modifies rules move recursive clauses to right
python data_labeler.py --level "s3" --catch_errors

# s1 has not errors
#python data_labeler.py --level "s1"