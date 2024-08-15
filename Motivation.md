# Bachelor project - motivation
By: Joshua Victor Niemelä and Mustafa Hekmat Al-Abdelamir
## Introduction
To alleviate various limitations of Graph Neural Networks for use in Graph Representation Learning, Topological Deep Learning is emerging as a promising field of machine learning. 
TDL has shown promising results in tackling over-smoothing and over-squashing, still, perhaps because of the young nature of the field, there is limited theoretical foundations for quantifying these possible advantages.
This poses a problem in making quantitative comparisons between various architectures for GRL, in this paper we attempt to lay the foundations for various metrics to provide insights on over-squashing and over-smoothing and how they relate to the model's ability to learn.
To verify and support our theoretical foundations we also run benchmarks against other approaches used for learning on graph data.

<!--
SIDENOTE 1: GNNs not based on message passing might not have the same problems of oversquashing oversmoothing
-->








## Data
We will be using the QM9, Benchmark dataset for graph classification and the Human Body segmentation datasets to benchmark our own implementations against other models cited in various papers.

Additionally, we will be constructing synthetic datasets to compare the expressivity between various models for the task of identifying various topological structures in graphs.

## Methods
We will be using Pytorch Geometric to build our models and to replicate models cited in other papers as a comparison and empirical exploration of GRL.

## Learning Objectives
1. Gain a solid understanding of the theoretical foundations of GNNs and their shortcomings such as over-smoothing and over-squashing.
2. Gain insights and understanding about TDL, and how topology is leveraged for learning.
3. Construct generalisable metrics to quantify various geometric and topological properties of GNNs and the datasets they are trained on.
4. Construct a model, for instance a transformer, that can learn topological features in data and benchmark against non-topological approaches.



## References




# old shit
## Learning Objectives
1. Understand the theory behind Graph Neural Networks (GNNs) and how they can be used to model graph-structured data.
2. Understand how topological information can be used to improve the performance of GNNs.
3. Construct a transformer and ordinary GAT model and compare their performance and training times. Here we hope to see that the GAT gets outperfomed by the transformer in performance and the GAT will be faster in training / more stable, there should then using graph-rewiring be some trade-off where a rewired GAT with additional edges approaches the performance of the transformer whilst not having increased too much in complexity, the converse should also apply that there should be edges that can be stripped away on a transformer (if it is seen as a fully-adjacently connected GAT) without significant degradation in performance
5. Investigate the oversquashing problem in GNNs and how it affects the ability for the model to learn long-range dependencies between nodes.
6. Investigate various methods utilising graph rewiring or other non-destructive methods such as Graph Echo State Networks to improve the performance of GNNs on long-range dependencies.
