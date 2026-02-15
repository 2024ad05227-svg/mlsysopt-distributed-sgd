# Distributed Mini-batch SGD (ML System Optimization)

This repository contains an implementation of synchronous data-parallel
mini-batch Stochastic Gradient Descent (SGD) using PyTorch Distributed
Data Parallel (DDP).

## Parallelization Strategy
- Synchronous Data Parallelism
- AllReduce-based gradient aggregation

## Files
- train_ddp.py : Distributed training script
- model.py : Model definition
- utils.py : Utility functions
- requirements.txt : Python dependencies

## Execution
The code is designed to be executed using PyTorch's distributed launcher:
