#!/bin/bash

# source /root/miniconda3/bin/activate myconda
# CUDA_VISIBLE_DEVICES=0,1 python ddp_tutorial.py
# conda deactivate

CUDA_VISIBLE_DEVICES=0,1 conda run -n myconda python ddp_tutorial.py