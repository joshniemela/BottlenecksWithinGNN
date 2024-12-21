#import "@preview/bamdone-aiaa:0.1.2": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge, shapes

#show: aiaa.with(
    title: "A unifying framework for quanitifying the data propogation bottlenecks of graph representation learning methods",

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
        name:"Raghav Selvan",
        job:"Assistant Professor, Supervisor",
        department:"DIKU"
      ),
    ),
    abstract: [[INSERT ABSTRACT WHEN POSSIBLE]]
)

#outline()

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

$ upright(bold(x))_i = theta sum _(j  in  cal(N)(i)  union i) frac(e_(j comma i), sqrt(hat(d)_j hat(d)_i)) upright(bold(x))_j $ <GCN_AGG_WITH_NORMALIZE>

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
    		let θ = 270deg - i*360deg/nodes.len()
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
sum _(forall V in G) & = theta sum_(j in cal(N)(1) union {1}) upright(bold(x))_j, theta in RR \
& = theta(x_1 + x_2 + x_3) quad "note that " x_1 = 0 \
& = theta(x_2 + x_3) arrow.r.double.long theta = 1
$ <three_node_solution>


Next, we do the same for the normalised example, $hat(d_i)$ is the number of neighbours of node $i$ plus one. This means, we have $hat(d)_2 = hat(d)_3 = 1$ and $hat(d)_1=3$.

$
upright(bold(x))'_1 & = theta sum_(j in cal(N)(i) union {i}) frac(1, sqrt(hat(d)_j hat(d)_i)) upright(bold(x))_j \

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

We observe that both models successfully learned the optimal parameters. However, it is important to acknowledge that the problem was contrived, and the task was directly analogous to the underlying mechanisms of the GCN. Consequently (summation of node features), the problem did not necessitate the use of an embedding. This simplification allowed for a clearer demonstration of the GCN's operational principles, yet it may not fully capture the intricacies involved in more nuanced applications.


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

Hence we opted to use a SAGE model [CITATION], this is Similar to the GCN model but with an additional weight parameter for the self-loop. We also use a bias parameter. We got an accuracy of $approx 50.9%$. // aber why do we give it a bias parameter?
This is not much better than the GCN model.  [THIS IS DEPENDENT ON EPSILON; EXPLAIN WHY]

== Five-nodes
TODO: blah blah we say how we generalised it to more nodes and it still worked



= "Tree neighbors-match"


== Introduction
In the paper by Alon et al. (2021) @alon_bottleneck_2021, the authors investigate the impact of modifying the last layer of Graph Convolutional Networks (GCNs) to be fully adjacent, meaning that it connects all nodes in the graph such that any node can send a message to any other node with just one intermediary node. This modification is posited to enhance the model's ability to capture global information from the graph, thereby improving its performance on various tasks such as node classification and link prediction. The authors provide empirical evidence demonstrating that this architectural change consistently leads to better results across different datasets and benchmarks. As part of this study, they construct a graph in which they claim to demonstrate the issue of over squashing. In this section, we aim to replicate their findings to validate their practical demonstration of over-squashing and the effectiveness of the fully adjacent last layer in GCNs.

== The dataset
The dataset of a given depth $d$ is comprised of $n$ unique and i.i.d. perfect binary trees of depth $d$. Each leaf in the tree has two features, a class, and the number of "leaves". We can represent a tree as a directed and connected graph $G in (V, E)$ where $(i, j) in E$. As a twist to the original paper, we index the nodes in level-order traversal, which has the nice property of allowing a simple definition for the adjacency matrix. 
We let $A$ represent the adjacency matrix for a given tree without any self-loops. Let $ A_(i,j)= 1 "if" floor(i/2) = j "else" 0. $
The reason for this is relatively simple. since the level-order traversal fills each layer in the graph before populating the next, we know that node $i$ must be on layer $floor(log_2(i))$.
//TODO: SOURCE???
We use self-loops since they have a positive effect on performance SOURCE??? 
//TODO:[CITE THE CURVATURE PAPER(?)]
: $overline(A)=A+I$, where $I$ is the identity matrix.

The $V in NN^(2 times n)$ contain our attributes for the nodes. All nodes other than nodes at depth $d$ have a class of 0: $V_(1, i_(1)) = 0, i_(1) in {0, 1, 2, ..., n-2^(d)-1, n-2^(d)}$. The remaining nodes are labelled in ascending order: $V_(1, i_2)= i_2-(n-2^d), i_2 in {n-2^d+1, ..., n-1, n}$.
The root has a random number of leaves between $1$ and $2^d$. The nodes in the last layer are sampled to have a random number of leaves between $1$ and $2^d$ without replacement. All nodes between depth $1$ to $d-1$ are set to have $0$ leaves. The label for the dataset is then finally set to be the class whose leaves match the number of leaves of the root. Note that the edge matrix is 1-indexed and the attribute matrix is 0-indexed for ease of notation.
// TODO: Fancy graph example?


== Experiment reproduction
Although in @alon_bottleneck_2021, the authors present their findings on the tree neighbours-match dataset, they do not provide results of implementing the fully adjacent last layer which they otherwise more widely propose as a heuristic approach to deal with over-squashing. We are going to refer to this method as "Last-FA" from  now on.
This has led us to implement the fully adjacent last layer and compare the model with and without Last-FA.
=== Model architecture
// TODO: redo this experiment???
The two node features are embedded in a 32 dimensional space using a linear layer with trainable weights without a bias parameter. We used RELU as our activation function and mean as our graph convolution aggregator. The models have \(d+1\) layers, where \(d\) is the depth of the trees in our given dataset.
We use the [INSERT THE NAME OF THIS THING] normalisation as utilised in PyTorch Geometric. We used ADAM and a reduce LR on plateau scheduler with [PARAMS?].
The last fully adjacent layer connects every single node to the root, we can omit the remaining pairwise connections since the resulting messages don't get propagated to the root before we finish the message passing. Let $r$ be the root node index, we then have: $E_("FA") subset.eq {(i, j) bar i in V, j = r}$.



We have in total run 1135 experiments, with a random distribution of parameters, to see how the GCN models with and without the fully adjacent last layer compare. 
// TODO: add this result back in, is it even valid anymore?
//The results are presented in Figure \ref{fig:assets/tree_experiment_graph}. More detailed results can be found \href{PLACE SOMETHING HERE MAYBE A LINK TO THE RESULTS IN THE}{here}.

== Discussion
In the code that the original authors provided for this experiment, they interestingly didn't report on the results of the Last-FA method on the Tree neighbors-match dataset. Testing on this dataset yielded poor results.



// They done done it Don, then they did damn did it didnt they?





