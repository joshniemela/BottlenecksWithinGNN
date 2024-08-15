# BachelorProject
This project will explore the oversquashing problem in Graph Neural Networks (GNNs) and how it affects the ability for the model to learn long-range dependencies between nodes.
The project will also explore the use of topological information in the form of the dataset's graph structure to improve the model's ability to learn long-range dependencies.  

Supervised by: Raghavendra Selvan  

## Random thoughts
Attention is analogous to a fully-adjacent graph layer, therefore it is a special case of a GNN.
Therefore, it should be possible to find some trade-off between losing computing efficiency and performance.

Jacobians and stuff show that node to node insensitivity is equivalent to oversquashing and we coudl use this to measure insensitivity in a TDL graph

Learnable topological representaiton.
measuring distances between relevant nodes.

replacing transformer attentoin with tdl based network


<!--
This might need to be moved to its own file or compiled to LaTeX
-->
# Bachelor project - motivation
By: Joshua Victor Niemelä and Mustafa Hekmat Al-Abdelamir
## Introduction

## Data
Various synthetic datasets or real datasets

## Methods

## Learning Objectives
1. Understand the theory behind Graph Neural Networks (GNNs) and how they can be used to model graph-structured data.
2. Understand how topological information can be used to improve the performance of GNNs.
3. Construct a transformer and ordinary GAT model and compare their performance and training times. Here we hope to see that the GAT gets outperfomed by the transformer in performance and the GAT will be faster in training / more stable, there should then using graph-rewiring be some trade-off where a rewired GAT with additional edges approaches the performance of the transformer whilst not having increased too much in complexity, the converse should also apply that there should be edges that can be stripped away on a transformer (if it is seen as a fully-adjacently connected GAT) without significant degradation in performance
5. Investigate the oversquashing problem in GNNs and how it affects the ability for the model to learn long-range dependencies between nodes.
6. Investigate various methods utilising graph rewiring or other non-destructive methods such as Graph Echo State Networks to improve the performance of GNNs on long-range dependencies.
