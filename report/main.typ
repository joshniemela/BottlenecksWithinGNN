#import "@preview/bamdone-aiaa:0.1.2": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge, shapes

#show: aiaa.with(
    title: "A unifying framework for quanitifying the data propagation bottlenecks of graph representation learning methods",

     bibliography: bibliography("sources.bib", style: "ieee"),
    authors-and-affiliations: (
      (
        name:"Mustafa Hekmat Al-Abdelamir",
        job:"Undergrad",
        department:"DIKU"
      ),
      (
        name:"Joshua Victor Niemelä.",
        job:"Undergrad",
        department:"DIKU"
      ),
      (
        name:"Raghavendra Selvan",
        job:"Assistant Professor, Supervisor",
        department:"DIKU"
      ),
    ),
    abstract: [[INSERT ABSTRACT WHEN POSSIBLE]]
)
#show link: underline

#outline(
  depth: 3,
  indent: auto,
)

= Introduction
Topological Deep Learning (TDL) is gaining traction as a novel approach @papamarkou_position:_2024 for Graph Representation Learning (GRL).
Leveraging topology, TDL has shown promising results in alleviating various limitations of Graph Neural Networks (GNNs) @horn_topological_2022
Two often-cited related @giraldo_trade-off_2023 shortcomings in GNNs are over-smoothing and over-squashing.
Over-smoothing occurs when individual node features become washed out and too similar after multiple graph convolutions @li_deeper_2018.
In message-passing, nodes send fixed-size messages to their neighbours, said neighbours aggregate the messages, update their features, then send out new messages and so on.
This process inevitably leads to information loss, as an increasingly large amount of information is compressed into fixed-size vectors.
This is known as over-smoothing @alon_bottleneck_2021.

Still, perhaps because of the young nature of the field, there is limited theoretical foundation for quantifying these possible advantages.
This poses a problem in making quantitative comparisons between various architectures for GRL.
In this paper, we attempt to lay the foundations for various metrics to provide insights on over-squashing and over-smoothing and how they relate to the model's ability to learn.
To verify and support our theoretical foundations, we also run benchmarks against other approaches used for learning on graph data @horn_topological_2022.

= Objectives

- Gain a solid understanding of the theory and application of Graph Convolutional Networks (GCNs).
- Investigate the over-squashing problem prevalent in graph representation learning and how it affects the ability of the model to learn long-range dependencies @alon_bottleneck_2021.
- Gain insights and identify various shortcomings of modern machine learning research.
- Shed light on what factors make a paper more or less reproducible.
- Establish a framework / template for what should be done to ensure that papers are as reproducible as possible.
// - Gain insights and understanding about TDL, how topology is leveraged for learning, and how it relates to the aforementioned bottlenecks @horn_topological_2022.
// - Construct generalisable metrics to quantify various geometric and topological properties of GNNs and the datasets they are trained on.
// - Construct a model, for instance, a transformer or CCNN @tdlbook, that can learn topological features in data and benchmark against non-topological approaches.


= Three-node GCN 

== Introduction

To dip our toes into Graph Neural Networks and eventually over-squashing, we started out with a simple GCN model with one layer and one input channel (each node has only one feature). Since the GCN aggregates all messages, including self-connections with the same coefficient, the GCN layer has exactly one parameter. The update and aggregate function with normalisation is described in eq @GCN_AGG_WITH_NORMALIZE, with $cal(N)(i)$ being the indices of the neighbours of the $i$'th vertex and $e_(j,i)$ being the edge weight between $j$ and $i$.

$ upright(bold(x))_i = theta sum _(j  in  cal(N)(i)  union {i}) frac(e_(j comma i), sqrt(hat(d)_j hat(d)_i)) upright(bold(x))_j $ <GCN_AGG_WITH_NORMALIZE>

The formula without normalisation in @GCN_AGG is identical, with the diagonal degree matrix term being omitted.

$ upright(bold(x))'_i = theta sum_(j in cal(N)(i) union {i}) e_(j, i) upright(bold(x))_j $ <GCN_AGG>

We construct a simple problem for our GCN, for which we initially analytically derive the optimal value for our parameter $theta$ both with and without normalization. Subsequently, we proceed to train the model and validate that the resulting weights align with our expectations. The underlying motivation for this approach is to formulate a straightforward Graph Neural Network (GNN) problem, which we solve analytically before employing numerical methods for model training. This dual approach facilitates a deeper understanding of the operational mechanics of the Graph Convolutional Network (GCN) model and elucidates the learning process inherent to it, as will be apparent further down in this report.

=== Regression problem
Our dataset / problem, $cal(D)={(G_1, y_1), ..., (G_n, y_n)}$ is a graph of three nodes $G_n in (V_n, E_n)$, with two children pointing into the root node $x_1$. For a given graph, we let the root node be set to 0, $x_1 = 0$. The children are sampled $~cal(U){0, 999}$. the target ($y$) is the sum of the entire graph $sum _(forall V in G)$. The readout is done by reading the root node value $x_1$.


// TODO: Node order is flipped dunno why
#let nodes = ($x_1$, $x_2$, $x_3$)

#let edges = (
	(0, 1),
	(0, 2),
)

#pad(y: 16pt,
  figure(
    diagram({
    	for (i, n) in nodes.enumerate() {
    		let θ = 90deg - i*360deg/nodes.len()
    		node((θ, 12mm), n, stroke: 0.5pt, name: str(i), shape: fletcher.shapes.circle)
    	}
    	for (from, to) in edges {
    		let bend = if (to, from) in edges { 10deg } else { 0deg }
    		// refer to nodes by label, e.g., <1>
    		edge(label(str(from)), label(str(to)), "<|-", bend: bend)
    	}
    }),
    caption: [The three nodes graph]
  ) 
) <three_nodes_graph>

Our edge weights are fixed to 1, aka. all edges have equal weighting, $e_(j, i)=1$ in eq @GCN_AGG.
With only one layer, we do just one graph convolution towards the root, hence we can disregard the updates for the two children nodes and only look at the update that affects the root, which means we can set $i=1$ as well in eq @GCN_AGG. This leaves us with a linear function with one parameter $theta$ as can be seen in eq @three_node_solution.


$
&theta sum_(j in cal(N)(1) union {1}) upright(bold(x))_j, theta in RR\
& = theta(x_1 + x_2 + x_3) quad "note that " x_1 = 0 \
& = theta(x_2 + x_3) arrow.r.double.long theta = 1
$ <three_node_solution>


Next, we do the same for the normalised example, $hat(d_i)$ is the number of neighbours of node $i$ plus one. This means, we have $hat(d)_2 = hat(d)_3 = 1$ and $hat(d)_1=3$.

$
upright(bold(x))'_1 & = theta sum_(j in cal(N)(1) union {1}) frac(1, sqrt(hat(d)_j hat(d)_1)) upright(bold(x))_j \

& = theta(frac(1, sqrt(hat(d)_2 hat(d)_1)) upright(bold(x))_2 + frac(1, sqrt(hat(d)_3 hat(d)_1)) upright(bold(x))_3) \

& = theta(frac(1, sqrt(1 dot.op 3)) upright(bold(x))_2 + frac(1, sqrt(1 dot.op 3)) upright(bold(x))_3) \

& = frac(theta, sqrt(3))(upright(bold(x))_2 + upright(bold(x))_3) arrow.r.double.long theta = sqrt(3) approx 1.732
$

For the two experiments, we got the resulting weights and losses after 15 runs with a learning rate of 0.1 and having trained for 100 epochs:
#align(center, block[
  #table(
    columns: (4em, auto, auto),
    table.header(
      [Base], [Median], [Std.]
    ),
    [Loss], [0.000744], [0.002407],
    [$theta$], [0.999976], [0.000032]
  )
  #table(
    columns: (4em, auto, auto),
    table.header(
      [Norm.], [Median], [Std.]
    ),
    [Loss], [0.000575], [0.001551],
    [$theta$], [1.732049], [0.000055]
  )]
)

We observe that both models successfully learned the optimal parameters. However, it is important to acknowledge that the problem was contrived, and the task was directly analogous to the underlying mechanisms of the GCN (summation of node features). Consequently, the problem did not necessitate the use of an embedding. This simplification allowed for a clearer demonstration of the GCN's operational principles, yet it may not fully capture the intricacies involved in more nuanced applications.


=== Classification problem
Next, we tried a classification problem on the same graph, which turned out to be slightly more complex. In this experiment we want to classify whether $x_1 = x_2+x_3$ is true or not for a given graph. $x_2, x_3$ are integers sampled from a uniform distribution in the range ${0,..., 9}$. Half of the generated graphs will have the property that $x_1 = x_2+x_3$ and the other half $x_1$ is a random integer in the range ${0,..., 18}$. We mark the target class after generation to account for the case that that $x_1$ is the sum of $x_2$ and $x_3$ also in the random case (there is a $1/19$ chance that the random case happens to also be a correct label). The distribution of classes will be as follows:

Let $X$ represent the sum of two independent 10-sided dice, where each die has faces numbered from 0 to 9. The possible values of $X$ range from 0 ($0+0$) to 18 ($9+9$). The probability distribution of $X$ forms a triangular shape because certain sums have more combinations than others.


// TODO: maybe dont make this a subsection?
==== PMF Definition
The PMF for $X$ is given by:

$
P(X = k) = min(k+1, 19 - k) / 100 "for" k in {0, 1, ..., 18}
$

Our $x_1$ is sampled independently and uniformly, hence the distribution will simply be $P(x_1 = k) = 1 / 19$

We get the joint distribution of $K=x_1$ by multiplying the distributions since they are independent

$
P(x_1 = k) = sum _(k in {0, 1, ..., 18}) P(X=k) dot P(x_1 = k) = frac(1, 19) sum _(k in {0, 1, ..., 18}) P(X=k)
$

Since we sum over all possible events in the distribution, the likelihood is 1 by definition. Hence we can simplify to:

$
P(K=x_1) = frac(1, 19)
$

Since sampling only happens on one half of our dataset, this means that our class distribution is 
$ P(y="true") = 1/2 - frac(1,19 dot 2) "and" P(y="false") = 1/2 + frac(1, 19 dot 2) $

We tried with the same GCN model, here we got an accuracy of $approx 0.4736$. Since this is approximately the same as the distribution of the classes, we can conclude that the model has not managed to learn the objective function.

In fact, we argue that this issue is not merely a matter of data or convergence but represents a fundamental limitation of the model. To determine whether $x_1 = x_2 + x_3$, the model must be able to process interactions between the messages (e.g., by subtracting $x_2 + x_3$ from $x_1$). However, since our GNN computes $z = w (x_1 + x_2 + x_3) + b$ as the node representation of $x_1$, it lacks the capacity to explicitly represent the subtraction or compare $x_1$ against $x_2 + x_3$. The shared weights ensure that each message contributes uniformly, pushing the node representation tensor in the same direction and thereby precluding the necessary relational reasoning.

We can analytically reason that giving $x_1$ a distinct weight $w_1$ and $x_2 + x_3$ a distinct weight $w_2$ and setting $-w_1 = w_2$, the node representation of $x_1$ will be:


$ z  &=  w_1 x _(1 ) +  w _(2 ) (x _2 +  x_3) +  b  \    &=  w_1 x_1 +  (-w_1)(x _(2 ) +  x _(3 )) +  b  \    &=  w _(1 ) (x _(1 ) - (x _(2 ) +  x_(3))) +  b $

When $x_1 = x_2 + x_3$, this simplifies to $z = b$, and more generally, we can find the difference between $x_1$ and $x_2 + x_3$ is given by:

$ lr(| z  -  frac(b ,w _(1 )) |) $

Hence we opted to use the PyTorch implementation of SAGEConv @hamiltonYL17, this is similar to the GCN model but with an additional weight parameter for the self-loop. We also use a bias parameter. We got an accuracy of $approx 50.9%$. // aber why do we give it a bias parameter?
This is not much better than the GCN model.  [THIS IS DEPENDENT ON EPSILON; EXPLAIN WHY]

// IMPORTED FROM LATEX
//[FORMALLY EXPLAIN THE PROBLEM]
//
//The models fail to learn since we have no non-linearity in the model. We need some behaviour
//that allows the model to learn that the further away from 0 the difference is,
//the more likely it is that the children nodes don't sum to the root node.
//Therefore, we want a function that maps 0 to 1 and values further away from 0 to 0. 
//For this, we simply chose the gaussian function \(e^{-x^2}\) which we
//appended to the readout of the model. With this we get perfect results with an
//accuracy of 100\%. On the downside, this model is very likely to not converge to an optimal solution: SHOW PLOT.
//
//We then used the function \(1-x^{2}\) which we composed with the readout. This was also able to get perfect accuracy but it was much more stable and only in rare cases it would result in a suboptimal solution of >95\%.
//accuracy of 100\%.
//
//Even still we notice that our model is really unstable, most of the time, it never converges on a solution. Since it only has two parameters, we can actually plot the loss function in 3D and analytically investigate the issue. As can be seen on figure \ref{fig:loss_surface}, there are two local minima, in the quadrant where the model parameters are both negative and positive. 
//
//\begin{figure}[H]
//  \centering
//  \input{assets/loss_3d.pgf}
//  \caption{The loss surface of the three-node graph classification problem.}
//  \label{fig:loss_surface}
//\end{figure}

// TODO: IS THIS NECCESARY TO KEEP
// Our tree neighbours using the approximation for birthday problem has almost 0 likelihood for collisions between test and train
// For each given depth \(d\), we have \(2^{d}! \cdot 2^{d}\) (\(2^{d}!\) permutations of the bottom layer, \(2^{d}\) possible root labels) possible trees / samples. We notice this means that for \(d=2\) and \(d=3\), we only get \(96\) and \(322560\) unique trees respectively.
// We sample from this dataset by generating a binary tree, then creating a permutation of the unique number of leaves and then randomly picking a class that the root should mimic.
// By the birthday paradox, we know this means these depths will very likely contain duplicate entries in the train and test data. Since they are both sampled IID this means it will not result in overfitting. In the event that the model has perfectly learnt the training data which contains all possible unique entries, we know that any future samples we throw at the model will also just contain the same entries we can classify.
// For depths greater then 3, the number of unique trees grows to the point where the likelihood of duplicate entries goes towards 0.





== Five-nodes
TODO: blah blah we say how we generalised it to more nodes and it still worked

= "Tree neighbors-match"


== Introduction
// TODO: mention that they also came up with the concept of oversquashign and that this should be a naive trivial solution that works
In the paper by Alon et al. (2021) @alon_bottleneck_2021, the authors investigate the impact of modifying the last layer of Graph Convolutional Networks (GCNs) to be fully adjacent, meaning that it connects all nodes in the graph such that any node can send a message to any other node with just one intermediary node. This modification is posited to enhance the model's ability to capture global information from the graph, thereby improving its performance on various tasks such as node classification and link prediction. The authors provide empirical evidence demonstrating that this architectural change consistently leads to better results across different datasets and benchmarks. As part of this study, they construct a graph in which they claim to demonstrate the issue of over squashing. In this section, we aim to replicate their findings to validate their practical demonstration of over-squashing and the effectiveness of the fully adjacent last layer in GCNs.

== The dataset
The dataset of a given depth $d$ is comprised of $n$ unique and i.i.d. perfect binary trees of depth $d$. Each leaf in the tree has two features, a class, and the number of "leaves". We can represent a tree as a directed and connected graph $G in (V, E)$ where $(i, j) in E$. As a twist to the original paper, we index the nodes in level-order traversal, which has the nice property of allowing a simple definition for the adjacency matrix. 
We let $A$ represent the adjacency matrix for a given tree in the dataset. Let 
$ A_(i,j)=
cases(
  1 "if" i=j, 
  1 "if" floor(i/2)=j, 0 "else")
$

The reason for this is as follows:
For a given node $n$, its parent node is determined by the position of $n$ on the layer. This position is calculated as the index of the node minus the sum of the number of nodes in all previous layers, or $n - sum _(i in [0 ... log_2(n)-1])$. To find the corresponding node in the previous layer, which has half as many nodes, we divide this position by 2 and round down (hence the addition of 1/2 and use of the floor function 
// TODO: fix this ugly sum notation
$floor(frac(n - sum _(i in [0 ... log_2(n)-1]) 2^i + 1, 2))$). To find the index of this new position, we must add the number of nodes that came before this layer, or $sum _(i in [0 ... log_2(n)-2]) 2^i$. When we write out and simplify the expression, we get:


$ "parent"(n) &= sum _(i in [0 ... log_2(n)-2]) 2^i + floor(frac(n - sum _(i in [0 ... log_2(n)-1]) 2^i + 1, 2)) \

 &= floor(frac(sum _(i in [1 ... log_2(n)-1]) 2^i + n - sum _(i in [0 ... log_2(n)-1]) 2^i + 1, 2)) \

 &= floor(frac(n - 2^0 + 1, 2)) \

 &= floor(frac(n, 2)) $

Note that we also add self-loops to the graphs since they have a positive effect on performance @topping_understanding_2022[p. 2].

The $V in NN^(2 times n)$ contain our attributes for the nodes. All nodes other than nodes at depth $d$ have a class of 0: $V_(1, i_(1)) = 0, i_(1) in {0, 1, 2, ..., n-2^(d)-1, n-2^(d)}$. The remaining nodes (on the last layer) are labelled in ascending order: $V_(1, i_2)= i_2-(n-2^d), i_2 in {n-2^d+1, ..., n-1, n}$.
The root has a random number of leaves between $1$ and $2^d$. The nodes in the last layer are sampled to have a random number of leaves between $1$ and $2^d$ without replacement. All nodes between depth $1$ to $d-1$ are set to have $0$ leaves. The label for the dataset is then finally set to be the class whose leaves match the number of leaves of the root. Note that the edge matrix is 1-indexed and the attribute matrix is 0-indexed for ease of notation.
// TODO: Fancy graph example?


== Experiment reproduction
Although in @alon_bottleneck_2021, the authors present their findings on the tree neighbours-match dataset, they do not provide results of implementing the fully adjacent last layer which they otherwise more widely propose as a heuristic approach to deal with over-squashing. We are going to refer to this method as "Last-FA" from  now on.
This has led us to implement the fully adjacent last layer and compare the model with and without Last-FA.
=== Model architecture
// TODO: redo this experiment???
The two node features are embedded in a 32 dimensional space using a linear layer with trainable weights without a bias parameter. We used ReLU as our activation function and mean as our graph convolution aggregator. The models have $d+1$ layers, where $d$ is the depth of the trees in our given dataset.
We use normalisation as implemented in PyTorch Geometric. We used Adam and a reduce LR on plateau scheduler with [TODO: PARAMS?].
The last fully adjacent layer connects every single node to the root, we can omit the remaining pairwise connections since the resulting messages don't get propagated to the root before we finish the message passing. Let $r$ be the root node index, we then have: $E_("FA") subset.eq {(i, r) bar i in V}$.

// TODO: is this needed
// Philosophy (hypothesis testing and duhem quine, blah blah)
//
// TODO: REVISIT EXPERIMENT
// The results of the tree neighbours-match experiment show that contrary to the findings in \cite{alon_bottleneck_2021}, the GCN with the fully adjacent last layer does not outperform the GCN model without the fully adjacent last layer. The results are consistent across all depths. This is an interesting result and motivates further investigation into the over-squashing problem on a more theoretical level rather than just heuristic, as it is difficult to draw any conclusions from our contradictory results alone.



We have in total run 1135 experiments, with a random distribution of parameters, to see how the GCN models with and without the fully adjacent last layer compare. 
// TODO: add this result back in, is it even valid anymore?
//The results are presented in Figure \ref{fig:assets/tree_experiment_graph}. More detailed results can be found \href{PLACE SOMETHING HERE MAYBE A LINK TO THE RESULTS IN THE}{here}.


== Alternative aggregators
The results of the tree neighbours-match experiment show that contrary to the findings in @alon_bottleneck_2021, the GCN with Last-FA does not outperform the GCN model without Last-FA. The results are consistent across all depths. This is an interesting result and motivates further investigation into the over-squashing problem on a more theoretical level rather than just heuristic, as it is difficult to draw any conclusions from our contradictory results alone. 

We theorise that the problem with learning the Tree-neighbors problem, is partly due to the aggregation method not being expressive enough.
To look further into this, we tried repeating the experiment using a multi-layer perceptron (MLP) as the aggregator.

//\[
//	h_v^{(k)} = \text{UPDATE}^{(k)} \left(MLP^{(k)}(\{ h_u^{(k-1)}: u \in \mathcal{N}(v) \cup v \}) \right)
//\]



== Discussion
// TODO: link to repo
In the #link("https://github.com/tech-srl/bottleneck/")[code that the original authors provided] for this experiment, they interestingly didn't report on the results of the Last-FA method on the Tree neighbors-match dataset. Testing on this dataset yielded poor results.




= Experiments to mitigate oversquashing

== Introduction
// TODO: write

// TODO: find name for this placeholder
== placeholder (Fully adjacent mean injection layer)
We suspect that over-squashing is caused by the fact that long range interactions are being 
watered out by the neighbours around the path of information flow. 
We therefore propose adding a global node to collect all the non-local information
from the graph. This will make so that the nodes have to send less non-local information
around the graph, resulting in less information that might be lost due to
over-squashing.

We do this by building on top of the SAGE framework @hamiltonYL17 and we simply add a special global node with its own set of weights (in this case, an arbitrary learnable function):
//TODO source????
$
x^(l+1)_i &= w^l_1 + w^l_2 dot "AGG"({x^l_j: j in cal(N)(i)})+f^l (x^l_i, x^l_G)+b^l\
x^(l+1)_G &= "AGG"({x^l_j: j in cal(V)})
$
In this setup, we have two trainable weights, one for the local aggregation, and one for the self loop. Furthermore we have a trainable bias. Lastly there is $f^l$, this is some trainable differentiable function, in our case we let it be a two-layer MLP with a hidden dimension of 512 and $tanh$ activation. The method of aggregation used in our experiments was mean aggregation, but any aggregator could be used.
The remaining normal GCN layers were set to have 128 hidden dimensions.

The global node is equivalent to performing a global aggregation of the graph.
We can then compute a weighting for information from the global node to the local nodes which is then added to each of the local nodes.

=== Experiment
To make our experiment as reproducible as possible, we chose to use Bayesian hyperparameter optimisation (BO), first randomly initialising with 5 points in our hyperparameter space, and then we searched using BO for an additional 15 new sets of parameters. Each parameter was then trained on once, and then the validation loss was used to select the best set. The validation loss was used as our objective, and we added a tiny penalty of $4 dot 10^(-7) dot "epochs"$, the motivation behind this additional penalty term was merely to reduce training times since very similar performance but significantly fewer epochs would be a preferred solution in this optimisation problem.

The parameters that were searched across were:
$log_10 ("epochs") in [2, 3.5[$, $log_10 ("learning_rate") in [-4, -1[$ and $log_10 (C) in [-4, -2 [$.
$C$ was used as the weight_decay argument in the PyTorch implementation of Adam, this is equivalent to applying L2 regularisation.


We performed this search across hidden layer depths 1 through 3 and with our TODO: placeholder method on the last layer and not on the last layer.

Each of these configurations was then trained on and evaluated 15 or more times. Some sets of values yielded noisy results and were thus evaluated more to decrease the statistical uncertainty.


=== Results
We compared the two methods with 1 to 3 hidden layers, 1 hidden layer performed better across the board and we thus only plot that for the distributions, and the QQ plot contains all three layer depths.
The observations were truncated between the 25 to 100 quantile to remove subpar runs.

#figure(image("assets/combined_plot.svg", width: 90%),
caption: [Comparison between the TODO: placeholder and base models.]
)

We chose to report the results using Q-Q plots and density plots. Summary statistics like the average, and standard deviation do not provide a lot of useful information when we are working with distributions that are non-normal, such as the resulting distributions from our experiments. Model A might outperform model B on average, but model B might rarely be better than any observation from A. This distinction might be relevant with the CiteSeer experiment where it is not immediately clear which model performs better.

The TODO: placeholder method performed significantly worse for Cora, mostly similar but slightly worse for Citeseer, and significantly better for PubMed. On Citeseer it can be noted that the single best observation is using TODO: placeholder, more runs must be performed to rule out that TODO: placeholder could be better but harder to train. 




== Stochastic Discrete Ricci Flow (SDRF)
The authors of @topping_understanding_2022 propose a method for improving graph representation learning via curvature based rewiring. With this they also hope to mitigate the effects of oversquashing.

Their method can be summarised as a stochastic sampling of edges and subsequent sequential mutations of the graph are performed until convergence is reached.

To gain more insights into oversquashing, we want to reproduce the results of this paper and then build ontop of the results. 
This is where we hit various hurdles with reproducing the results.
The authors provide the reader with the relevant code used for the experiment on #link("https://github.com/jctops/understanding-oversquashing")[GitHub]. The code is undocumented and does not closely follow the mathematical abstractions used in the paper. The implementation is a seemingly efficient implementation relying on some undescribed linear algebra optimisations. We were unable to show that the code implemented in Python was correct and represented the same algorithm presented in the paper. They also use adjacency matrices instead of sets which might imply that this implementations wouldn't scale to large graphs. TODO: remove this claim maybe?

We then began writing up our own implementation closely following the paper, which meant we had to first compute the Ricci curvature between two edges.

=== The naive implementation
To compute the Ricci curvature, we must compute three non-trivial values for a given edge $i~j$. The number of 3-cycles, the number of 4-cycles without diagonals, and the maximum number of times a specific node is visited with a 4-cycle from i to j ($gamma_max$).

We implemented this entirely using sets, this is a more naive approach without any optimisations. 
For the 3-cycles ($\#_triangle (i, j)$) we noticed that this was equivalent to finding a mutual neighbour of i and j. 
// This was implemented as a set union 

The 4-cycle sets ($\#_square (i, j)$) were more complicated to compute. 
Bad nodes for the edge $i~j$ are defined as ${i, j} union \#_triangle (i,j)$.
We start off by finding the neighbours of $i$, then remove the bad nodes. These will be our first set of candidate nodes.
For each of these candidate nodes, we then found their neighbours, removed the bad nodes again, and then added the nodes that yielded a path to $j$ (there exists an edge from this neighbouring candidate and $j$).
The $gamma_max$, was found by computing the 4-cycles on $i~j$ and counting how many times a neighbouring node was included in the cycle. This was then also done for $j~i$.

We chose to not explore this implementation further due to computational constraints and moved on to trying to replicate the results using the author's optimised implementation.

=== Code replication
We tried replicating the algorithm the authors provided on GitHub alongside the paper. 
Here we do not consider the correctness of the algorithm, and merely sought to replicate the algorithm and see if their implementation was significantly more efficient than our naive implementation.

We made one difference which was to use a sparse representation of our matrices to ensure that large datasets could be put into memory (datasets like PubMed would not fit into VRAM).

This implementation took $approx 153.0947$ seconds to do compute the Balanced Forman Curvature on all edges of PubMed (88648 edges) with a speed of $approx 579.04$ edges a second. This was done on a Ryzen 5 2600 and the algorithm was implemented in Julia.

=== Runtime analysis
The biggest bottleneck in the SDRF algorithm is computing the Ricci or Balanced Forman Curvature (BFC) @topping_understanding_2022[p. 4]:
$
"Ric"(i, j) &:= 2/d_i + 2/d_j - 2 + 2 (|\#_triangle (i, j)|)/(max{d_i, d_j})
+ 2 (|\#_triangle (i, j)|)/(min{d_i, d_j})\
&+ 2 ((gamma_max)^(-1))/(max{d_i, d_j})(|\#_square^i (i, j)|+|\#_square^j (i, j)|)
$
Let $B_r(i)$ be the set of nodes reachable from $i$ within $r$ hops.
We need to recompute the BFC for all edges in $B_2(i) union B_2(j)$ every time we add or remove an edge $i ~ j$.
This is because the BFC can be affected by any node within radius 2 of the edge $i~j$. // TODO: is this edge or node radius of 2?

When computing the improvement vector $x$ for a given edge, we need to compute the BFC for all edges possible between $B_1(i) times B_1(j)$. This can easily become quite a large number of computations if either of the nodes happen to have a large number of neighbours.

This quickly becomes infeasible for dense graphs or large graphs due to cache utilisation. Our implementation has no efficient way of storing graphs so that neighbouring nodes have any guarantee to be in cache alongside other neighbours. This is not a trivial problem to solve since most graphs have no natural ordering of nodes or edges. The computation thus becomes very inefficient for the computations of 4-cycles and the degeneracy.

Furthermore, for a lot of graph datasets, the cardinality of the receptive field grows exponentially (for PubMed, the average degree is $approx 4.469$ and the maximum degree is 171). This means the algorithm risks having a very high number of nodes that need to be searched for each single computation of the curvature. This is further accentuated in datasets where there are a few nodes with a large number of neighbours, since you are very likely to hit one of these nodes and then having to search through all of the neighbours.

For our naive approach, we found marginal improvements to computational performance by caching previously computed curvature values for each edge in the graph and then re-updating them every time an edge is added or removed. This improvement doesn't tackle the more fundamental problem of the algorithm essentially having to do a large number of random accesses across the graph.

The authors of @giraldo_trade-off_2023 also report not being able to reproduce the model performance of SDRF, also due to BFC being too expensive to compute efficiently. They then substitute this metric with their own Jost-Liu Curvature (JLC) which yields worse results (comparable to the baseline GCN) than the SOTA results that @topping_understanding_2022 reported. The GitHub repository as of 2024-12-22 also has an #link("https://github.com/jctops/understanding-oversquashing/issues/3")[open issue] from July 15, 2022 where multiple users report not being able to reproduce the same positive results from the paper. 

We chose to not investigate this algorithm further due to the aforementioned issues.


== TOGL (TOpological Graph Layer)
// Performance wise we notice that the Persistance diagram computation runs the same, when single threaded, on both a low end cpu with just 3MB og cache and a midrange cpu core with 16MB cache. 
//This confirms our suspicion that the random ordering of nodes in the filtration process, is maximally not spacially local. This is because of the uniform sampling of the nodes across the datastructure which stores the graph. 
//Although the filtration could in principle be computed with high effeciency, in practice, the computation is done from the very slow system RAM and only sequencially, since the algorithm requires information from the previous step to compute the next step. This makes it inherently disadvantagous to algorithm with high spacial locality and algorithms relying on vectorizable operations.



// TODO: sub-conclusion of this section but not the whole paper
== Discussion (takeaways from experiments)
// Write conclusions and takeaways from
// the experiments and what went wrong

=== Reflection on experimental methodology and other shortcomings
Where do we go from here? TOGL, FA and SDRF all propose a solution but
not where to continue researching from there.

=== Mismatch between maths and implementation
//Implementation details might be unmotivated and up to personal interpretation.
//the mathematical implementation might be intractible or not be the most efficient implementation
//without implementating unintuitive or non-trivial optimisations.

// TODO: find name for this
== A framework for more reproducible benchmarks and experiments

Paper and motivations of code should be easily transferable to other languages and frameworks
This also includes synthetic datasets
For instance, caching, dealing with infinity
Code quality is bad (Essentially prototyped code in notebooks and the sorts are used for the final paper)
this is analogous to deploying untested prototypes in production environments.

Machine learning methodology prioritises acquiring a good model, but not reproducible models or behaviour. (professional vs academic)


A lot of papers prioritise the mathematics of the models and theory but often brush over the implementation details.
These implementation details are essential to the reproducibility of the experiments and the results.
Without these details, it's often only possible to reproduce the results using the exact same code and hardware that the authors used. 
This leads to the problem of dependency hell where the code is not portable and the results are not reproducible.
This problem is partly due to the majority of papers within the field of machine learning using Python.

CITE SDRF. In this paper, they propose a new algorihtm for graph representation learning, 
but the naive approach (if you just directly convert the maths to code) is intractible and 
it is not feasible to train a model with this approach.
The authors provide their own code in the form of a github repository.
This code might work, but we failed to get the dependencies to resolve and have the code run.
Their specific implementation uses some efficient linear algebra tricks that are not
documented in the code nor the paper, which means its not immediately apparent
that this code even is correct according to the mathematical motivation in the paper.

A lot of code has the characteristics of being swiftly prototyped code, with 
little to no documentation, no tests and no focus on portability.
Often the code is also not modular which makes it hard to reuse the code in other 
reproductions or experiments, and you end up having to rewrite the code from scratch.

A way to mitigate these problems, is to better document implemetation details that
deviate from the mathematics, and to provide model weights, and datasets. 
If you have the capacity to train on dozens of GPUs then you can also provide
the datasets and model weights in a portable and reproducible format.

=== Dependency hell and reproducibiity woes in Python}
Python has notoriously bad package management, and the common workflow is to use virtual environments which may persist state across sessions or experiments. To make matters worse, there are multiple
standards for virtual environments, all of which are not compatible with each other.
The common workflow is also to use notebooks which are not very friendly to version control, which
makes it difficult to compare different versions of the same experiment.
For package management you can use things like Pip, Poetry, Pipenv, Conda, uv and more. Most of these solutions are not compatible with each other. 
Some packages also cause trouble with reproducibility, for instance Numpy or other packages that depend on C++ bindings. 
Lastly another problem with Python's ecosystem is that packages quickly become abanadoned and break
due to external factors without any changes having been done to the package itself.
These bindings are often not explicit dependencies and might break on containerised systems or non-standard operating systems such as NixOS or systems without preinstalled C++ libraries.

A concrete example could be that the paper [CITE ALON AND YAHAV] which both use an old
version of numpy (there has since been made breaking changes to Numpy), TensorFlow, and PyTorch all
in the same experiment. There are also very few comments in the repository.
Without the high level implementation details, it is quite difficult to reproduce the results.

Outdated dependencies / breaking changes
Dead dependencies / Missing dependencies
No documentation or poor documentation




== placeholder
Hyperparameters must be chosen so that they are orthogonal (andrew ng) to each other.
Heuristic and empirical approaches such as early stopping or SGD with restarts are good for 
finding a good set of hyperparameters as quickly as possible, but they are not
reproducible nor do they provide the academic community with 
any significant insights into the performance of the model or the data.

Instead of early stopping which affects (andrew ng blah blah bias and variance)
we can use a more robust approach such as Bayesian Optimisation (BO) which 
is an informed search on the hyperparameter space that requires
less iterations and computation time than random search or grid search.
Bayesian Optimisation also allows the search space to be less constrained and biased by
arbitrarily selected priors.

Most papers (source) report the results as either the best result or the average across
multiple runs with a standard deviation. These two summary statisticss are
not as informative as plots on the different distributions of models.
Other plots such as Q-Q plots or candlesticks would also be more informative, since
the best model is typically what we are interested in producing, but the maximum of a sample of models is a statistically meaningless number which cannot be compared to other distributions.
Standard error on observations

One set of models might on average be worse, but for instance still have
a high volume of samples that can significantly outperform the other set.

Documenting how the model is trained is important, but often the weights are not
published alongside the paper which makes it impossible to reproduce the exact 
test results, an author might have published the results of an exceedingly
rare model that yields SOTA results in which the results are not reproducible by
other researchers.

Algortihmic replicability is also a concern, papers might propose some mathematically
elegant solution to a problem, but the implementation itself is either not 
reproducible, giving the wrong results, or a naive approach is intractible.
It is therefore important to document the algortihmic optimisations that were
used to arrive at the results, be it caching, algebraic simplifcations, or
other tricks. (source SRDF and jost / liu saying that they cant compute SDRF)

Various levels of reproducibility issues:
algortihmic replicability: algortihm either is not proven to be
correct or motivated to be correct, the implementation is not
avaialble, reproducible or intractible.
learning replicability: the method used to train the model is not reproducible.
data replicability: data is not accesible or not possible to reproduce.
methodology replicability
== Reproducibility
REMOVE?

== NOTES (REMOVE)
TODO: NOTE ABOUT MEAN INJECTION LAYER: Might be better to do multiple validation trains per searched param to get an average loss before moving on to the next one
=== bayesian search thoughts
Multiple runs within the objective function?: yields an average which goes against our idea of allowing param configurations that might be unstable in training but yield good model performance
Multiple bayesian searches: Maybe not neccesary?
Cross validation???: Yes no?

// They done done it Don, then they did damn did it didnt they?



// TODO LIST
// 1. Epsilon explanation on three nodes - Mustafa

// 2a. Tabular data report for three_nodes_classification - Mustafa
// See summary.jl in three_nodes_regression
// See train.py, rewrite similar `save_runs` function so we get CSV and models saved
// 2b. Reexport loss surface plots (mby plots.jl if we have time) as svg 

// 3. Do the same for five-nodes - Mustafa

// Tree neighbors: The awful one
// 4a. Redo tree-neighbors-match with bayesian optimisation and the same setup as in
// graph_benchmark folder (main.py & summary.jl)
// optimise on log_epochs, log_learning_rate, weight_decay
// sweep through all combinations of depth (2-5), with/without Last-FA
// - Mustafa

// 4b. Visualise the data - Josh 
// 4c. Repeat 4.a with MLP aggregation
// 4d. Josh visualises stuff (again) - Josh
// 4e. Write discussion for tree neighbours - Josh/Mustafa

// 5. TOGL - Mustafa

// 6. what is todo:placeholder: fa to be called - Josh/Mustafa

// 7. Write discussion for all experiments

// 8. ML Framework - To be assigned

// 9. Choice of template (line spacing, numbering etc)
