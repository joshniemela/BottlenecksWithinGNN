# Potentially Useful Sources
## [Universal Graph Neural Networks without Message Passing](https://openreview.net/forum?id=P0bfBJaD4KP)
## [Non-convolutional Graph Neural Networks](https://arxiv.org/abs/2408.00165)
## [On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology] (https://arxiv.org/abs/2302.02941)
## [Topological Deep Learning](https://tdlbook.org/introduction)
This is a useful book about TDL

## [Position: Topological Deep Learning is the New Frontier for Relational Learning](https://arxiv.org/pdf/2402.08871)
This is a meta-analysis of the field of topological deep learning. It is a good starting point for understanding the field.  
Section 6.1 talks about the advantages of topological deep learning and is a section Raghav is interested in exploring further.

## [On the Bottleneck of Graph Neural Networks and its Practical Implications](https://arxiv.org/abs/2006.05205)
This paper talks about the oversquashing problem and long-range interactions in GNNs. Networks such as GCN and GIN that do not differentiate between messages perform worse than GATs and GGNN.  
SOTA models were beaten by adding a trivial fully-adjacent graph layer to the end of the GNN.  
The authors also made a video explaining the paper: [YouTube](https://www.youtube.com/watch?v=XQHfkHTAo0s)  
This was a source suggested by Raghav.

## [Topological Graph Neural Networks](https://arxiv.org/abs/2102.07835)
To come  
This was a source suggested by Raghav.

## [Understanding over-squashing and bottlenecks on graphs via curvature](https://arxiv.org/abs/2111.14522)

## [Topology-Informed Graph Transformer](https://arxiv.org/abs/2402.02005)

## [Hierarchical message-passing graph neural networks](https://link.springer.com/article/10.1007/s10618-022-00890-9)

## [Leave Graphs Alone: Addressing Over-Squashing without Rewiring](https://openreview.net/forum?id=vEbUaN9Z2V8)

## [Understanding Oversquashing in GNNs through the Lens of Effective Resistance](https://arxiv.org/pdf/2302.06835)
The authors found out that over-squashing is a different problem from "under-reaching", which is that the network isn't large enough to be able to make a node reach another node. They also reproduce the results of adding a FA making all GNNs better. They found out that the number of edges added from the FA also correlates with a reduction in errors even if it isn't the entire layer. Hidden sizes are much less relevant than just adding the FA.

## [Hierarchical Graph Representation Learning with Differentiable Pooling](https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf)

## [Hierarchical Graph Neural Networks](https://arxiv.org/pdf/2105.03388)

## [On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks](https://arxiv.org/pdf/2212.02374)

## [Position: Topological Deep Learning is the New Frontier for Relational Learning](https://arxiv.org/abs/2402.08871)

## [Masked Attention is All You Need for Graphs](https://arxiv.org/pdf/2402.10793)
