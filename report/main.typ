#import "@preview/bamdone-aiaa:0.1.2": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge, shapes
#import "@preview/plotst:0.2.0": *
#import "@preview/ouset:0.2.0": ouset, overset
#import "@preview/subpar:0.2.0"

#set page(numbering: "1 of 1")
#show link: underline

#show: aiaa.with(
  // or Bridging the gap between theory and implementation: 
    title: "Bottlenecks and reproducibility in Graph Neural Networks: A Study of Challenges and Mitigations",

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
    )
)
#outline(
  depth: 2,
  indent: auto,
)
#set par(spacing: 1.5em)

= Introduction
The rapid advancement of artificial intelligence (AI) has been significantly shaped by the widespread adoption of encoder-decoder transformer architectures, which have revolutionized various domains, including natural language processing, computer vision, and computational biology. These architectures, characterized by their ability to model relationships through self-attention mechanisms, despite their impressive capabilities, often require vast amounts of labeled data and extensive computational resources, raising concerns about their scalability and generalization in real-world applications, as well as accessibility of state of the art AI.

In light of these challenges, We want to shift our focus towards model architectures that incorporate relationships as inductive biases, particularly graph neural networks (GNNs) which previous research suggests has architectural advantages in generalizability @yang2023graphneuralnetworksinherently. GNNs are designed to operate on data structured as graphs, consisting of entities (nodes) and their relations (edges), giving the designer the ability to control the propagational flow of information. 

Despite the promising capabilities of graph neural networks (GNNs), several bottlenecks and challenges hinder their widespread adoption and effectiveness. Two often-cited related @giraldo_trade-off_2023 shortcomings in GNNs are over-smoothing and over-squashing. Over-smoothing occurs when individual node features become washed out and too similar after multiple graph convolutions @li_deeper_2018. Meanwhile over-squashing happens when an increasing number of node messages are cramped into a single fixed size vector node representation, after several message passing rounds @topping_understanding_2022.

There are also various other, more or less trivial, bottlenecks. In this project we will be exploring and documenting these bottlenecks while trying to gain a deeper understanding by analytically describing and, if possible, mitigating each issue we encounter. We also introduce a novel way to view over-smoothing, as a factor of doubled information, which attempts to explain why it happens even at lower layer counts. As an experimental mitigation of over-squashing we will try to embed topological features into node representations, specifically persistent homologies as described in @horn_topological_2022. We will also propose a novel way of compressing the path lengths between nodes by introducing highly connected nodes in an architecture we call HubGCN, to theoretically accommodate for shallower GNNs with fewer layers and hence less predisposition to over-smoothing and squashing while still incorporating long range dependencies.

// TODO: rephrase objectives since htey do not align with the project
= Objectives

- Develop a comprehensive understanding of the theory and practical applications of Graph Convolutional Networks (GCNs).
- Examine the causes and implications of the over-smoothing problem in graph representation learning.
- Explore the over-squashing phenomenon in graph representation learning, focusing on its impact on a model's ability to learn long-range dependencies @alon_bottleneck_2021.
- Acquire the skills to implement and evaluate solutions from existing research, assessing their performance and optimization potential.
- Investigate and apply topological methods, such as persistent homology and topological embeddings @horn_topological_2022, to mitigate over-squashing in Graph Neural Networks.
- Gain a deeper understanding of the current challenges and trends in machine learning research.  
- Identify and analyze the factors that contribute to or hinder the reproducibility of machine learning studies, particularly in the context of the reproducibility crisis @Gundersen_Kjensmo_2018.

// TODO: remove these?
// - Establish a framework / template for what should be done to ensure that papers are as reproducible as possible.
// - Gain insights and understanding about TDL, how topology is leveraged for learning, and how it relates to the aforementioned bottlenecks @horn_topological_2022.
// - Construct generalisable metrics to quantify various geometric and topological properties of GNNs and the datasets they are trained on.
// - Construct a model, for instance, a transformer or CCNN @tdlbook, that can learn topological features in data and benchmark against non-topological approaches.


= Three-node GCN Regression <regression-problem>

== Introduction 
// TODO: Raghav doesnt like the casual language
As an initial step towards exploring Graph Neural Networks (GNNs) and addressing the associated bottlenecks, we begin with a basic implementation of a Graph Convolutional Network (GCN). This model consists of a single layer, scalar node features, and a minimal number of parameters. The primary objective is to construct a straightforward problem for the GCN, for which the optimal solution can be analytically derived based on the parameter $theta$, both with and without normalization. Following this, the model is trained using PyTorch, and the resulting weights are evaluated to ensure alignment with the expected outcomes. This dual approach facilitates a deeper understanding of both the theoretical underpinnings of message-passing neural networks and the practical aspects of utilizing Python libraries for training machine learning models on graph-structured data.


== Background
The simplest form of a Graph Convolutional Network (GCN) is one that aggregates features from all neighboring nodes, including a potential self-connection, using the same aggregation coefficient. This type of network captures the essence of convolutional operations on graphs, where the node's features are updated based on the information from its local neighborhood. We begin our exploration with this foundational network design, as it provides a straightforward yet insightful introduction to the principles of GCNs.

The update and aggregation process for this GCN, incorporating normalization, is described mathematically in Equation @GCN_AGG_WITH_NORMALIZE. Here $cal(N)(i)$ denotes the set of indices corresponding to the neighbors of node $i$, and $e_{j,i}$ represents the weight of the edge connecting node $j$ to node $i$. To account for self-connections, we define $overline(cal(N))(i)=cal(N)(i) union {i}$, which extends the neighborhood of $i$ to include itself.

With normalization applied, the aggregation formula is expressed as @GCN_AGG_WITH_NORMALIZE:

$ x'_i = theta sum _(j  in  overline(cal(N))(i)) frac(e_(j comma i), sqrt(hat(d)_j hat(d)_i)) x_j $ <GCN_AGG_WITH_NORMALIZE>

In this equation, $hat(d)_i$ represents the degree of node $i$ (including self-loops), and normalization ensures that the contribution of each node is appropriately scaled, mitigating the effects of varying node degrees.

For comparison, the aggregation formula without normalization, as described in Equation @GCN_AGG, omits the diagonal degree matrix terms:

$ x'_i = theta sum_(j in overline(cal(N))(i)) e_(j, i) x_j $ <GCN_AGG>



== Method <threenodesregressionmehtod>
The dataset and problem setup, $cal(D)={(G_1, y_1), ..., (G_n, y_n)}$, consist of graphs $G_n in (V_n, E_n)$, each comprising three nodes. In every graph, two child nodes are directed towards a root node, denoted as $x_1$. The root node is initialized to 0, such that $x_1 = 0$. The values of the child nodes are sampled from a uniform distribution, $~cal(U){0, 999}$. The target value ($y_n$) for each graph is defined as the total sum of the node values: $y_n = sum_(v in V_n) v$.

To evaluate the model, the readout operation is conducted by observing the updated value of the root node, $x'_1$, after a single round of message passing. For this setup, the theoretically expected solution is $x'_1 = x_2 + x_3$.

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
    caption: [Graph of a simple three-node structure. The root node $x_1$, with two child nodes, $x_2$ and $x_3$, connected to it via directed edges. The directed edges indicate the flow of information during the message-passing process in the Graph Convolutional Network (GCN).]
  ) 
) <three_nodes_graph>

In this experiment, we assign all edge weights a constant value of $e_(j, i)=1 thick forall (i, j) in V times V$. This simplification ensures that all contributions from neighboring nodes are treated equally during aggregation, eliminating potential confounding effects of varying edge weights. Since the dataset is synthetic, any choice of edge weights would be arbitrary either way.

We begin by deriving an analytical solution that the Graph Convolutional Network (GCN) could theoretically produce to solve the given problem. Following this, we proceed to train the GCN and evaluate whether the model successfully reproduces the analytical solution through numerical optimization.

== Experiment

Since the model consists of a single layer, only one round of graph convolution is performed, and the message-passing will pass information from the child nodes to the root node. Consequently, updates to the two child nodes can be disregarded as we are only interested by the representation in $x_1'$. Thus, we focus exclusively on the update affecting the root node and set $i=1$ in @GCN_AGG, where $e_{j, i} = 1$, we derive a linear function parameterized by $theta$, as presented in Equation @three_node_solution


$
x'_1 &=theta sum_(j in overline(cal(N))(1)) x_j,thick theta in RR\
& = theta(x_1 + x_2 + x_3) quad "note that " x_1 = 0 \
& = theta(x_2 + x_3) arrow.r.double.long theta = 1
$ <three_node_solution>


Next, we perform the same derivation for the normalized case, where the degree $hat(d_i)$ of a node $i$ is defined as the number of its neighbors plus one (to account for self-loops). For our specific setup, this results in $hat(d)_2 = hat(d)_3 = 1$ and $hat(d)_1=3$.

$
upright(bold(x))'_1 & = theta sum_(j in overline(cal(N))(1)) frac(1, sqrt(hat(d)_j hat(d)_1)) upright(bold(x))_j \

& = theta(frac(1, sqrt(hat(d)_2 hat(d)_1)) upright(bold(x))_2 + frac(1, sqrt(hat(d)_3 hat(d)_1)) upright(bold(x))_3) \

& = theta(frac(1, sqrt(1 dot.op 3)) upright(bold(x))_2 + frac(1, sqrt(1 dot.op 3)) upright(bold(x))_3) \

& = frac(theta, sqrt(3))(upright(bold(x))_2 + upright(bold(x))_3) arrow.r.double.long theta = sqrt(3) approx 1.732
$ <threenodesregressionanalyticalsolutionwithnorm>

To empirically validate this analytical solution, we conducted two sets of experiments: one for the unnormalized case and another for the normalized case. Both experiments involved training the GCN over 15 independent runs using a learning rate of 0.1 for 100 epochs. The resulting weights and losses were recorded to assess the model's ability to converge to the derived optimal parameters.


#subpar.grid(
  figure(
    table(
      columns: (4em, auto, auto),
      table.header(
        [Base], [Median], [Std.]
      ),
      [Loss], [0.000744], [0.002407],
      [$theta$], [0.999976], [0.000032]
    ), caption: [w/o normalization]
  ),
  figure(
    table(
      columns: (4em, auto, auto),
      table.header(
        [Norm.], [Median], [Std.]
      ),
      [Loss], [0.000575], [0.001551],
      [$theta$], [1.732049], [0.000055]
    ), caption: [w/ normalization]
  ),
  columns: 2,
  caption: [The results of training on the problem presented data presented in @threenodesregressionmehtod. We report the parameter value after the training, which seem to be consistent with the results in @three_node_solution and @threenodesregressionanalyticalsolutionwithnorm.],
  label: <threenoderegressiontable>
)


== Results

Our empirical analysis reveals that both models have successfully converged on the optimal parameters, thereby demonstrating that our conceptual understanding wich underlies @three_node_solution and @threenodesregressionanalyticalsolutionwithnorm is sound.

== Discussion

Notwithstanding the successful performance of the models, it is essential to acknowledge the contrived nature of the problem. The task was deliberately designed to be directly analogous to the fundamental mechanisms of a Graph Convolutional Network (GCN), specifically the summation of node features. As a result, the problem did not necessitate the use of a non-linear embedding, which would have introduced additional complexity and nuance. This simplification, while facilitating a clear demonstration of the GCN's capabilities, may not fully capture the intricacies and challenges inherent in more complex and real-world applications.

== Conclusion 

[idk]



= Three-node GCN Classification

== Introduction

We conduct a classification experiment on the same graph, which presents a moderate increase in complexity. The objective of this experiment is to classify whether the relationship $x_1 = x_2 + x_3$ holds true for a given graph. In this context, $x_2$ and $x_3$ are integer values sampled from a uniform distribution within the range ${0,..., 9}$. The generated graphs are then divided into two categories: in half of the cases, the relationship $x_1 = x_2 + x_3$ is satisfied, while in the remaining half, $x_1$ is assigned a random integer value within the range ${0,..., 18}$. To account for the possibility that $x_1$ might coincidentally equal the sum of $x_2$ and $x_3$ in the random case, we explicitly label the target class after generation. Notably, there exists a $1/19$ probability that the randomly assigned label will coincidentally correspond to the correct classification, as demonstrated in @appendix-1.


We tried with the same GCN model as in @regression-problem. Note that we wish the result to be $x'_1 approx 0$ for $x_1=x_2+x_3$ and $x'_1 approx 1$ for  $x_1 eq.not x_2 + x_3$#footnote[We define $1$ as the positive class, and $0$ as the negative class.] but we don't care if $x'_1$ is positive or negative as whether $x'_1$ is smaller or bigger than $x_2+x_3$ is irrelevant, we just care about whether the equality is true. This will become relevant later when we want to do a readout. With our setup based pm the one in @regression-problem, we got an accuracy of $approx 0.47$. Since this is approximately the same as the distribution of the classes $(1/2- 1/2 1/ 19 approx 0.473)$, we can conclude that our model failed to learn the objective function.

We argue that this represents a fundamental limitation of our model. To determine whether the statement $x_1 = x_2 + x_3$ is true, the model must be able to process interactions between the messages (e.g., by subtracting $x_2 + x_3$ from $x_1$), not merely summing them together. 
// TODO: we introduce a bias term, do we also do this in our experiment? we have discrepency with whats written above
However, since our GNN computes $x'_1 = theta_1 (x_1 + x_2 + x_3) + theta_0$, it lacks the capacity to explicitly represent a comparison between $x_1$ against $x_2 + x_3$. The shared weights ensure that each message contributes uniformly, always pushing the node representation in the same direction.

We can analytically reason that by giving $x_1$ a distinct weight $theta_1$ and $x_2 + x_3$ a distinct weight $theta_2$ and setting $-theta_1 = theta_2$, that we get the following node representation of $x'_1 = x_1^c$:


$ x_1^c  &=  theta_1 x _(1 ) +  theta _(2 ) (x _2 +  x_3) +  theta_0  \    &=  theta_1 x_1 +  (-theta_1)(x _(2 ) +  x _(3 )) +  theta_0  \    &=  theta _(1 ) (x _(1 ) - (x _(2 ) +  x_(3))) +  theta_0 $

When $x_1 = x_2 + x_3$, this simplifies to $x^c_1=theta_0$, and more generally, we can find that the difference between $x_1$ and $x_2 + x_3$ is given by:

$ lr(| x_1^c  -  frac(theta_0 ,theta _(1 )) |) $ <three_nodes_class_analytical_solution>

// TODO: positive and negative is again being funky???
Hence with the correct choice of $theta_1$ and $theta_2$ and $theta_0=0$ (no bias), that $x'_1$ directly provides the difference between $x_1$ and $x_2+x_3$. This observation implies that $x_1^c = 0$ should be classified as a positive instance, while $x_1^c eq.not 0$ warrants a negative classification.

We opted to use the PyTorch implementation of SAGEConv @hamiltonYL17, which is similar to the GCN model but with an additional weight parameter for the self-loop.

Now if we train the model it would still not converge on a solution. This is because our hidden representation $x_1^c$ needs to somehow be mapped to a real number bounded within $[0, 1]$ and our mapping needs to be continuous, and reflect that smaller $|x_1^c|$ is closer to 1 and larger $|x_1^c|$ is closer to 0. The natural choice here is to use a gaussian mapping $g(x) = e^(-x^2)$. We find that sometimes we converge to the perfect solution $theta_1=-theta_2$ but not always. In fact it seems that the model is unstable, because of local minimals in the loss landscape, see @threenodesclassgaussianlosslandscape. [TODO: WHY ARE THERE LOCAL MINIMAS?]. 

To tackle this instability, we explored a simpler alternative mapping: $1 - x^2$, which has a much simpler derivative. Although this mapping does not guarantee that $x_1^c$ is mapped to the interval $[0,1]$, instead providing a mapping to $]-infinity, 1]$, it is suitable for our classification task, since we interpret the model output, by assigning it to the closest class numerically. In our case since we are using classes 0 and 1, This is equivalent to thresholding the values from the interval $[0,1]$, specifically at $1/2$. This simpler mapping generates a concave loss function, leading to a substantial improvement in how stable the model is under training, with the model successfully converging to the perfect solution almost every time.

#figure(
  image("assets/three_neighbour_classifier_gaussian_activation.svg", width: 75%),
  caption: [Loss surface of SAGE model on Three-Nodes-classification data with a gaussian mapping]
) <threenodesclassgaussianlosslandscape>

#figure(
  image("assets/three_neighbour_classifier_simple_activation.svg", width: 75%),
  caption: [Loss surface of SAGE model on Three-Nodes-classification data with a quadratic mapping]
)

We note though that we have made an analytical solution to the classification problem. 
It would still be more proper to train the model end to end without us having to manually provide a motivated mapping function, since this defeats the purpose of training a model. 

We therefore repeated the experiment with a trainable readout.
We have an MLP with 1 hidden layer using `ReLU` activation and a hidden dimension of 2.
This should be enough to approximate our mapping. This is because we want to approximate the function in @desired_readout which consists of just two linear functions, selectively chosen at two intervals. Note that the function can be "stretched" and "squeezed" about the x-axis since we are simply interested in the y axis intersection at $x=plus.minus 1/2$.

#let data = (
  (-1.5, -2), (0, 1), (1.5, -2)
)

#let x_axis = axis(min: -2, max: 2, step: 0.5, helper_lines: true, location: "bottom", title: "x")
#let y_axis = axis(min: -2, max: 2, step: 0.5, helper_lines: true, location: "left", title: "y" )
#let pl = plot(data: data, axes: (x_axis, y_axis))

#graph_plot(pl, (50%, 25%), caption: "Desired readout function for the
three-nodes classification problem") <desired_readout>

We train the model, but we do not converge on a correct solution. To make sure that our analytical reasoning is not at fault, we manually inject known good values for the parameters in the GNN and train just the MLP parameters. Although the training is still very unstable, we do sometimes converge on a correct solution. Hence the problem here is in our training setup.
// TODO: figure out why this happens? or just remove the section about this being unstable
// this is most likely a python quirk
// We should probably remove this yes, but it is not a python quirk, the loss landscapes are legitimately different
We are in using the `CrossEntropyLoss` [This is a problem but I don't know why]. We switched to `BCELoss` with logits (so that we may avoid compressing our MLP range to $[0,1]$ arbitrarily). This nets out good results, the accuracy distribution of our trained models can be seen on @accuracy_results_mlp2-2-bcelosslogit.

// NOTE:
// We should write about how the model fails to converge more often than it even converges
// Its quite interesting  that such a simple problem is likely to 
// fail to converge
// 
// Could it be because the dataset being discrete and thus the gradient might become 0 at a lot of places?


// lambda is set to 0.01 (bandwidth) and we truncate in Q2 to Q4
#figure(
  image("assets/kde-mlp2-2-bcelosswlogits.svg", width: 50%),
  caption: [Estimate for distribution of solution accuracies when training our GNN on the three nodes classification problem.]
) <accuracy_results_mlp2-2-bcelosslogit>


When the GNN is converging on the correct solution (100% accuracy) the MLP readout function is also converging roughly on the expected solution as can be seen on @actualtrainedreadoutout

#figure(
  image("assets/output_layer_mlp2-2-bcelosswlogits.22073192-edc3-46fa-8948-84fb3d1fdb0d.svg", width: 50%),
  caption: [The readout function trained numerically (and MLP with 2 hidden dims and 1 hidden layer, with `ReLU` activation).]
) <actualtrainedreadoutout>


== Discussion

The experiments conducted with the three-node GCN is a telling exploration of the limitations, bottlenecks, and issues associated with the simplistic implementation of a graph convolutional network. Below, we summarise the key challenges encountered in our exploration.

// TODO: rephrase title --- How about now?
//=== Lack of Relational Reasoning because of shared weights
In the classification problem, the model failed to learn the objective function, given a linear mapping. The task of determining whether $x_1 = x_2 + x_3$ requires the ability to compare $x_1$ against $x_2 + x_3$, which the shared weights and homogenous aggregation could not facilitate. This highlights a fundamental limitation: a GCN as implemented, cannot explicitly represent or process non-trivial relations between nodes in a single message pass. This is quite concerning as one of the strengths of GNNs are the use of shared weights which reduce computational complexity, makes it possible to use the same layer on each node and its neighbors no matter the degree of the node, and facilitates the learning of common relational logic in the graph. 

// TODO: similar to before, remove the thing with crossentropy vs bce
//=== The Need for a Proper and Expressive Readout
The classification problem highlighted the necessity for a trainable readout mechanism to map hidden representations to target outputs. While introducing an MLP improved the mapping, its training remained unstable, especially under `CrossEntropyLoss`. Switching to `BCELoss` with logits yielded better results, but the instability persisted, indicating a need for further refinement in the training setup.

//=== Non-convex loss landscapes
The loss landscape of the GCN with a Gaussian mapping contained local minima that impeded convergence. Transitioning to a simpler mapping, such as $1 - x^2$, improved stability but introduced arbitrary thresholding choices and restricted options for loss functions—preventing the use of BCE, which would have been the most suitable choice. Furthermore, increasing the number of parameters or activations generally disrupted the stability of numerical optimization methods.

//=== Class imbalance in data
The slight imbalance in the class distribution (stemming from the probabilistic nature of $x_1 = x_2 + x_3$) posed an additional challenge in the evaluation of our experiments. We needed to know the exact distribution of our classes, so that when the model’s accuracy matched the class distribution baseline, we could conclusively say that it failed to learn the target task beyond random guessing.

// TODO: This section is very short, remove?
== Limitations

The regression task was straightforward, and the GCN successfully learned the optimal parameters. However, this success stemmed from the fact that the task mirrored the inherent aggregation mechanism of the GCN. While useful as a pedagogical exercise, the oversimplification did not stress-test the model’s capabilities in complex scenarios. As for the classification task, although it is similarly contrived, we think it represented well some of the basic issues and pitfalls of GNNs even when there is just one layer.


= Five-nodes regression (under-reaching experiment)

This experiment is similar to the three-node experiment, but we now have five nodes instead of three. The difference being that the problem radius is now 2, hence we will need 2 hidden layers to propagate the information from the leaves to the root. We now introduce new notation, instead of $x'_i$ representing the value of $x_i$ after message passing, we let $x^l_i$ denote the value of $x_i$ after $l$ rounds of message passing (with 0 being the original value).

#pad(
  figure(
    diagram({
      let nodes = ($x_5$, $x_4$, $x_1$, $x_2$, $x_3$)
      let edges = (
        (3, 2),
        (4, 3),
        (1, 2),
        (0, 1),
      )
      for (i, n) in nodes.enumerate() {
            node((i,0), n, stroke: 0.5pt, name: str(i), shape: fletcher.shapes.circle)
          }
          for (from, to) in edges {
            let bend = if (to, from) in edges { 10deg } else { 0deg }
            // refer to nodes by label, e.g., <1>
            edge(label(str(from)), label(str(to)), "-|>", bend: bend)
          }
        }),
    caption: [The five nodes graph]
  )
)

The information flow in this graph is illustrated on @5nodes2layer. 

#figure(
  diagram({
    let nodes = ($x_5^0$, $x_4^0$, $x_1^0$, $x_2^0$, $x_3^0$, $x_5^1$, $x_4^1$, $x_1^1$, $x_2^1$, $x_3^1$, $x_5^2$, $x_4^2$, $x_1^2$, $x_2^2$, $x_3^2$)
    let edges = (
      // (3, 2),
      // (4, 3),
      // (1, 2),
      // (0, 1),
      (3+5, 2+5),
      (4+5, 3+5),
      (1+5, 2+5),
      (0+5, 1+5),
      (3+10, 2+10),
      (4+10, 3+10),
      (1+10, 2+10),
      (0+10, 1+10),
      (3, 3+5),
      (4, 4+5),
      (1, 1+5),
      (0, 0+5),
      (2, 2+5),
      (3+5, 3+5+5),
      (4+5, 4+5+5),
      (1+5, 1+5+5),
      (0+5, 0+5+5),
      (2+5, 2+5+5),
      (3, 2+5),
      (4, 3+5),
      (1, 2+5),
      (0, 1+5),
      // (3, 2+5+5),
      // (4, 3+5+5),
      // (1, 2+5+5),
      // (0, 1+5+5),
      (3+5, 2+5+5),
      (4+5, 3+5+5),
      (1+5, 2+5+5),
      (0+5, 1+5+5),
    )

    let layer_colors = (green, red, blue)
    
    for (i, n) in nodes.enumerate() {
          let pos = calc.rem(i, 5)
          let ypos = calc.floor(i/5)
          node((pos,ypos), n, stroke: 0.5pt, name: str(i), shape: fletcher.shapes.circle)
        }
        for (from, to) in edges {
          let from_layer = calc.floor(from / 5)
          let to_layer = calc.floor(to / 5)
          let color = if from_layer == to_layer { layer_colors.at(from_layer) } else { black }
          let bend = if (to, from) in ((0,16),(4,12)) { 10deg } else { 0deg }
          // refer to nodes by label, e.g., <1>
          edge(label(str(from)), label(str(to)), "-|>", bend: bend, stroke: color)
        }
      }),
  caption: [The five nodes graph 2 GNN layers. Red messages are passed on the first layer, while blue messages are passed on the second layer. At the bottom is the output layer. The black arrows illustrate the information flow between layers]
) <5nodes2layer>

We are doing a readout from node $x_1$, which after message passing of the first layer will have the representation:
$ x_1^1 = gamma^(1) (x_1^(0), plus.circle (phi.alt^(1) (x_1^(0), x_2^(0),e_(2, 1)),phi.alt^(1) (x_1^(0), x_4^(0),e_(4, 1))) $

Where $plus.circle$ is an aggregator (sum in our case), and $gamma$ and $phi.alt$ are differentiable functions.

Clearly the information in $x_5^0$ and $x_3^0$ is never passed and hence the node $x_1$ will not be able incorporate information from these nodes. Hence we get an accuracy of virtually 0% on our benchmarks as can be seen on @fivenodesbenchmarks.

#figure(
  image("assets/five-nodes.svg", width: 60%),
  caption: [Result on Five nodes dataset given the layer count. Model output has been rounded to nearest integer and after which accuracy has been calculated.]
) <fivenodesbenchmarks>

But at the second layer, the information from $x_2^2$ and $x_4^2$ is passed to $x_1^2$. We may consider the following message passes:

$ x_2^2 = gamma^(2) (x_2^(1), plus.circle (phi.alt^(2) (x_2^(1), x_3^(1),e_(2, 3)),phi.alt^(2) (x_2^(1), x_1^(1),e_(2, 1))) $
$ x_4^2 = gamma^(2) (x_4^(1), plus.circle (phi.alt^(2) (x_4^(1), x_5^(1),e_(4, 5)),phi.alt^(2) (x_4^(1), x_1^(1),e_(4, 1))) $ 

We keep note of every node representation which has been passed to these two nodes, and then look at the representation of $x_1^2$ which receives messages from them:

$ x_1^2 = gamma^(2) (x_1^(1), plus.circle (phi.alt^(2) (x_1^(1), x_2^(1),e_(1, 2)),phi.alt^(2) (x_1^(1), x_4^(1),e_(1, 4))) $ 

We note that all the nodes are present in this message. Hence information from the entire graph is passed to $x_1^2$ and we may read from this node. We see on @fivenodesbenchmarks the model can converge, and in fact, does so quite often.

== Discussion

Under-reaching is a rather intuitive bottleneck, in such a way that we don't really feel the need for this experiment to prove or understand it. But for consistency we have benchmarked our contrived problem, and we have seen the expected results.

== Limitations

Out of time constraints we have not made an analytical solution before running the algorithm, which means our testing hypothesis is not super well defined, or isolated. It is our belief that the observed bottleneck is under-reaching, and in fact we analytically show that it is present, but we do not show that it is the only bottleneck present and the only bottleneck alleviated in the two testing cases. By adding an additional layer, we change several factors, crucially including the parameter count.

= Over-squashing & Over-smoothing bottlenecks

Two often cited issue in the literature are over-squashing and over-smoothing 

// use definitions for these two phenomena in the literature and report on it

// def of over-smoothing
Over-smoothing refers to the phenomena of node representations becoming more similar the more layers you stack on a GNN, hence making it more difficult to distinguish and classify each individual node. Since over-smoothing refers to the similarity of node representations in the network, naturally any similarity measure between nodes could be used to measure the phenomena, but previous research suggests that a more trivial measure such at mean average distance, which is sometimes used, can be deficient for asymptotic analysis of over-squashing @rusch2023surveyoversmoothinggraphneural[p.3] and furthermore Dirichlet energy is suggested as the better measure currently in use in the literature:

$
cal(E)(upright(bold(X))^n) = sum_(i in cal(V)) sum_(j in cal(N)_i) lr(|| frac(upright(bold(X))_i^n,sqrt(1+d_i)) - frac(upright(bold(X))_j^n, sqrt(1+d_j)) ||)_2^2
$ <over-smoothing-metric>
Where n is the layer. This is essentially a normalized sum over the square euclidian distances between nodes at layer n.

Over-squashing seems to be a related issue @giraldo_trade-off_2023, but which is defined more by the inability of a single tensor to sufficiently represent the information @papamarkou_position:_2024

// page 5 of https://arxiv.org/pdf/2212.10701 might be interesting. When we go to a deeper layer, we get messages from all our neighbouring nodes again, but our receptive field grows slower, hence the new nodes become fractionally fewer, and add virtually no information.

Although @over-smoothing-metric gives us the opportunity to measure over-smoothing, it doesn't by itself tell us much about why it exists. Some literature suggests that over-smoothing is actually the result of the of "running out" of new nodes to pass to any particular node, or rather that the proportion of the absolute number of new nodes to already seen nodes in a given message disproportionate even at lower layer counts @wu2023nonasymptoticanalysisoversmoothinggraph[p.5]. 

We propose that over-smoothing is actually caused or extenuated by duplicate messages in the message passing process which is asymptotically more significant than the raw number of nodes introduced in a message pass. In a given simple graph $overset(circle,arrow.turn.b) <- overset(circle,arrow.turn.b) <- overset(circle,arrow.turn.b)$, we notice that the information in the 2nd node is passed to the 1st node in the first round of message passing. At the next layer the message received by the 1st node will contain information from node 2 and 3, since node 3 sent information to node 2 in the first layer, which is now being propagated further. But each node is not given the same significance in node 1. In fact there is a bias induced by the fact that the information in node 1 has more paths coming from node 2 on layer 2 than to node 3. Hence the message in node 2 is "duplicated" and fed to node 1 in the message passing process. In this case this might be a desirable effect as intuitively we might intend for the information in node 2 to be more relevant to node 1 than node 3 is. But this effect is least pronounced in a graphs where nodes degree is uniform such as in our toy example, and actually gets significantly more intrusive in natural graphs with highly connected nodes.

Before we introduce the theory behind our suggestion, we will present an empirical result which hopefully gives an intuition behind our motivation for this approach. In @nodegrowthfig we see that in normal graphs there is often a growth phase early on, where the number of new nodes at each layer is growing very fast and makes up a significant portion of the nodes in the message passing round. This runs contrary to the suggestion in @wu2023nonasymptoticanalysisoversmoothinggraph which states that over-smoothing is can be explained as shallow depths by the proportionally small amount of new nodes for de-noising.

#subpar.grid(
  figure(image("assets/nodes-per-radius-0-cora.svg"),caption: [Cora]),
  figure(image("assets/nodes-per-radius-0-pubmed.svg"), caption: [PubMed]),
  figure(image("assets/nodes-per-radius-1-citeseer.svg"),caption: [CiteSeer]),
  columns: 3,
  caption: [The growth of nodes in the receptive field of various datasets.],
  label: <nodegrowthfig>
) 

In @nodegrowthratefig we see that the new nodes make out a relatively large proportion of all the nodes in the receptive field, and then rapidly disappear as there are no more connected nodes to be encountered in the (sub)graph.

#figure(
  image("assets/receptive-field-grow-node-1.svg", width: 50%),
  caption: [The growth rate of new nodes in the receptive field of various datasets.]
) <nodegrowthratefig>

== The exorbitant privilege of the well-connected node

As an alternative but closely related hypothesis we introduce the exorbitant privilege of the well-connected node. 

We use the matrix power of the adjacency matrix $upright(bold(A))^n$ to count the number of paths from any given node to another with number of message passing rounds/layers $n$. We specify that the nodes in radii $r$ from node $x_i$ will be those where $upright(bold(A))_(x_i)^r >= 1$. And to find the nodes which are exactly $r$ steps away from $x_i$ we have:

$ cal(V)_r := {x_i in cal(V) : delta(upright(bold(A))^r_x_i, 0) - delta(upright(bold(A))^(r-1)_x_i, 0) = 1} $

Considering the number of paths $p_r^(x_i->x_j)$ at layer $r$ from $x_i$ to $x_j$ we know that $p_r^(x_i->x_j) : p_r^(x_i->x_j) gt.eq (p_(r-1)^(x_i->x_j))^2$ and we know specifically that when $p_(r-1) > 1$ the number of paths grows exponentially. This is because the number of paths for $x_i$ is given by the dot product of all outgoing paths of $x_i$ and the corresponding incoming paths (of the corresponding nodes) of $x_j$. Hence if there already is more than one path between $x_i$ and $x_j$, the cross product will at the very least include the term $p_(r-1)^(x_i->x_j) dot p_(r-1)^(x_i->x_j)$.

This begs the question; under what conditions will $p_r^(x_i->x_j) > 1$? In any condition where there are at least two paths leading from $x_i$ to $x_j$ (including direct paths with no intermediary nodes), or in other words, were the dot product of the incoming and outgoing paths sums to $gt 2$, which turns out to be a lot of the time if there is any path from $x_i$ to $x_j$ at all. This inherently advantages closer nodes which begin their path "grows spurt" early on, making them exponentially more represented in the message passing-scheme than other nodes. Additionally it especially advantages nodes with lots of connections, since the paths of these nodes will grow extremely fast in relation. 

In @coralayer1to5nodepathsloga and @coralayer6nodepaths look at what this means in practice we measure the number of paths from every node to an arbitrary chosen node, and show that a small percentage of nodes, get to pass off their information an overwhelmingly large amount of times even at shallow layer depths.

#grid(
  columns: 2,
  column-gutter: 5%,
 [#figure(
  image("assets/cora-paths-layer-1-5.svg"),
  caption: [Cora dataset. Already at 5 layers, some nodes send their information through more than a hundred paths, leading to duplicate information, while most other nodes have exponentially less paths to deliver their information. Notice that in this instance it is likely that the proportional difference is worse than exponential as can be seen by the curve even in logarithmic space.]
) <coralayer1to5nodepathsloga>], 
  [#figure(
    image("assets/cora-paths-layer-5-6.svg"),
    caption: [already at layer 6 the problem is ballooned out of hand. The non-logarithmic y-axis, extenuates the problem. This leads to an inherit bias in the network for information carried by  a few highly connected nodes.]
  ) <coralayer6nodepaths>]
)

This phenomena is even more accentuated by more connected graphs such as PubMed.


// page 5 of https://arxiv.org/pdf/2303.10993 gradient gating seems intersting. Can we employ it in the tree neighbours experiemnt to isolate the over-squasing effect?

// report on how we may measure the alledged factors which cause these two bottlenecks

// report on how prevelant they actual are, both in general and in relation to each other.



= "Tree neighbors-match"
// TODO: IS THIS NECCESARY TO KEEP
// Our tree neighbours using the approximation for birthday problem has almost 0 likelihood for collisions between test and train
// For each given depth \(d\), we have \(2^{d}! \cdot 2^{d}\) (\(2^{d}!\) permutations of the bottom layer, \(2^{d}\) possible root labels) possible trees / samples. We notice this means that for \(d=2\) and \(d=3\), we only get \(96\) and \(322560\) unique trees respectively.
// We sample from this dataset by generating a binary tree, then creating a permutation of the unique number of leaves and then randomly picking a class that the root should mimic.
// By the birthday paradox, we know this means these depths will very likely contain duplicate entries in the train and test data. Since they are both sampled IID this means it will not result in overfitting. In the event that the model has perfectly learnt the training data which contains all possible unique entries, we know that any future samples we throw at the model will also just contain the same entries we can classify.
// For depths greater then 3, the number of unique trees grows to the point where the likelihood of duplicate entries goes towards 0.

// TODO: mention that they also came up with the concept of oversquashign and that this should be a naive trivial solution that works
In the paper by Alon et al. (2021) @alon_bottleneck_2021, the authors investigate the impact of modifying the last layer of Graph Convolutional Networks (GCNs). More concretely, they implement a "Fully adjacent last layer", meaning that all nodes in the last layer are connected so that any node can send a message to any other node without any intermediate hops. This modification is posited to alleviate the over-squashing that might be present in our GCN model. This over-squashing affects the model's ability to capture global information from the graph, thereby reducing the performance on various tasks such as node classification and link prediction. The authors provide empirical evidence demonstrating that this architectural change consistently leads to better results across different datasets and benchmarks. As part of this study, they construct a graph dataset, "Tree Neighbors-Match", in which they claim to demonstrate the issue of over squashing. In this section, we aim to replicate their findings to validate their practical demonstration of over-squashing and the effectiveness of the fully adjacent last layer in GCNs.

== The dataset
The dataset of a given depth $d$ is comprised of $n$ unique and i.i.d. perfect binary trees of depth $d$. Each leaf in the tree has two features, a class, and the number of "leaves". We can represent a tree as a directed and connected graph $G in (V, E)$ where $(i, j) in E$. 
We let $A$ represent the adjacency matrix for a given tree in the dataset:
$ A_(i,j)=
cases(
  1 "if" i=j, 
  1 "if" floor(i/2)=j, 0 "else")
$
Note that we also add self-loops to the graphs#footnote("Implementation detail: self-loops are concatenated to the end of the edge-list after constructing the graph as a perfect binary tree.") since they have a positive effect on performance @topping_understanding_2022[p. 2].

As a twist to the original paper, we index the nodes in level-order traversal within an array, which has the nice property of allowing a simple definition for the edge list. This approach lets us avoid having to do the same recursive generation of the trees. We do this by filling the array with the following three functions: $"LEFT"(i)=2i$, $"RIGHT"(i)=2i+1$ and $"PARENT"(i)=floor(i/2)$ as defined by @Cormen2009-aq[p. 152].

The $V in NN^(2 times n)$ contain our attributes for the nodes. 
All nodes other than nodes at depth $d$ have a class of 0. The nodes on the last layer of the tree are labelled sequentially $1$ through $d$:
$ V_(1,i) = cases(
  0 &"if" i in {1, 2, ..., 2^d, 2^d-1}\
  i-2^d &"if" i in {2^d,... , n-2, n-1}
) $

The root, $V_(2,0)$, is sampled as a random number of leaves between $1$ and $2^d$. The nodes in the last layer 
$V_(2, i),thick i in {2^d, ..., n-1} $
are sampled to have a random number of leaves between $1$ and $2^d$ without replacement. All nodes between depth $1$ to $d-1$ are set to have $0$ leaves $(V_(2, i)=0, i in {1,...,2^d-1})$ The label for the dataset is then finally set to be the index of the leaf that matches the root.


// TODO: Fancy graph example?

== Experiment reproduction
Although in @alon_bottleneck_2021, the authors present the over-squashing bottleneck on the "Tree Neighbors-Match" dataset, they do not provide results of applying their fully adjacent last layer method to the dataset, which they widely propose as a heuristic approach to deal with over-squashing. We are going to refer to this method as "Last-FA" from now on.
This has led us to implement the Last-FA method and compare the model with and without Last-FA on the "Tree Neighbors-Match" dataset.

=== Model architecture
The two node features are embedded in a 32 dimensional space using a linear layer with trainable weights without a bias parameter. We used `ReLU` as our activation function and mean as our graph convolution aggregator. The models have $d+1$ layers, where $d$ is the depth of the trees in our given dataset.
We use normalisation as implemented in PyTorch Geometric.
The last fully adjacent layer connects every single node to the root, we can omit the remaining pairwise connections since the resulting messages don't get propagated to the root before we finish the message passing. Let $r$ be the root node index, we then have: $E_("FA") subset.eq {(i, r) bar i in V}$.

The results of the tree neighbours-match experiment show that contrary to the findings in @alon_bottleneck_2021, the GCN with Last-FA does not outperform the baseline GCN model. The results are consistent across all depths. 
// TODO: should this line be deleted since we didnt explore it theoretically?
This is an interesting result and motivates further investigation into the over-squashing problem on a more theoretical level rather than just heuristic, as it is difficult to draw any conclusions from our contradictory results alone.

// TODO: what are our hyper params? they are in wandb somewhere
//Furthermore, we also have a learning rate scheduler which has a factor of 0.9 and a patience of 20 epochs.
//We trained until the model was early stopped.
We have in total run 1000 experiments, with a random distribution of parameters, to see how the GCN models with and without the Last-FA layer compare:
#figure(
  image("assets/tree_neighbor_qq.svg", width: 80%),
  caption: [Comparison of accuracy between baseline and Last-FA grouped by depth and normalised between maximum accuracy for baseline at depth $d$ and random guess accuracy $(2^(-d))$.]
) <tree-neighbor-qq>
On @tree-neighbor-qq, we see that the Last-FA method is much more unstable on depth 3 but more stable on depth 4. Training on depth 3 frequently yields models worse than baseline, and rarely, a model that has an accuracy above 25% scaled by baseline accuracy.
For depths 4 and 5, we get very unstable training and worse performance overall.


We also tried to reproduce the same plot as @alon_bottleneck_2021 comparing Last-FA and baseline. We also test the baseline model with MLP aggregation (explained in @MLP-agg) as shown below:
#figure(
  image("assets/tree_neighbor_accuracy.svg", width: 60%),
  caption: [Comparison of maximum train accuracy between baseline, Last-FA and MLP aggregation as depth is increased.]
) <tree-neighbor-acc>

 On @tree-neighbor-acc we see the same overall trend with the baseline as in @alon_bottleneck_2021, where increasing the depth of the trees results in a significant reduction to accuracy whereas in depths 2 and 3, both Last-FA and the baseline manage to achieve an accuracy of 100%.
 For depths 4 and 5, we see the Last-FA method yielding different results than what is to be expected. Here we see that the Last-FA method performs significantly worse than expected, since Last-FA should alleviate over-squashing and `tree-neighbors-match` should be, by construction, a dataset which is heavily over-squashed @alon_bottleneck_2021.

// TODO:mmvoe the results to a place where they make sense
== Results

#figure(
  image("assets/kde_plots_by_tree_depth.svg"),
  caption: [TODO: title]
)


== More expressive aggregators <MLP-agg>
The results of the tree neighbours-match experiment show that contrary to the findings in @alon_bottleneck_2021, the GCN with Last-FA does not outperform the GCN model without Last-FA. The results are consistent across all depths. This is an interesting result and motivates further investigation into the over-squashing problem on a more theoretical level rather than just heuristic, as it is difficult to draw any conclusions from our contradictory results alone. 

We theorise that the problem with learning the "Tree neighbors-match" problem, is partly due to the aggregation method not being expressive enough.
To look further into this, we tried repeating the experiment using a multi-layer perceptron (MLP) as the aggregator in place of mean.
Here we weren't interested in the distribution of performance, we merely wanted to show the existence of model weights that are noticeably more performant than using a less expressive aggregator such as mean.

The model is defined as follows:
$
x_i^(l+1) = "UPDATE"^l ("MLP"^l ({x^l_j: j in cal(N)(i) union {i}} ))\
$

We trained from depths 2 to 5 with 40000 trees, an initial learning rate of 0.01, batch size of 2048 and an early stop patience 25 epochs.
The MLP aggregator is a 2-layer MLP with dimension of 96 with `ReLU` activation.
Furthermore, we also have a learning rate scheduler which has a factor of 0.9 and a patience of 20 epochs.
We trained until the model was early stopped and then we would restart with the best model weights and the learning rate set back to the initial of 0.01.

For depths 2 and 3, the GCN with MLP aggregation converged on the first try.
For depth 4, we had to restart it twice.
For depth 5, we set the batch-size to 8192 and had to restart the training three times until early stopping.

The results of using MLP aggregation can be seen on @tree-neighbor-acc. We see that the MLP manages to get a training accuracy of 99.7% for depth 4, and an accuracy of 39.1% for depth 5. This method exhibits very similar behaviour as @alon_bottleneck_2021 reported on GAT and GGNNs. @alon_bottleneck_2021 reported the bottleneck happening at one depth deeper for GGNN and GAT. We also note that the results are not entirely isolated to the removal of addition of MLP aggregation since we have a fixed set of hyper-parameters with learning rate scheduling and warm restarts#footnote("The learning rate is reset to the initial rate after the model early stops.") instead of doing a random search on the hyper-parameters as we did in the experiment comparing Last-FA and the baseline.

== Discussion
Alon and Yahav provide the code for their experiments on GitHub at
#link("https://github.com/tech-srl/bottleneck/").
They interestingly didn't report on the results of the Last-FA method on the "Tree neighbors-match" dataset.
Considering that this was the initial synthetic dataset they propose as an example of over-squashing, we found this to be a strange choice which lead us to test Last-FA on "Tree neighbors-match".

Looking through the code revealed that Alon and Yahav had in fact already written an implementation of Last-FA for use on "Tree neighbors-match", means they either chose not to report on the results, or they chose to not run the experiment.

The results we got by testing Last-FA on "Tree neighbors-match" were quite poor and could imply that the method doesn't generalise to as many datasets as would be implied.


= Experiments to mitigate over-squashing

In message-passing, nodes send fixed-size messages to their neighbours, said neighbours aggregate the messages, update their features, then send out new messages and so on.
// TODO: this description feels a bit intro'y? should it be moved to hte top of hte paper or be rephrased
This process inevitably leads to information loss, as an increasingly large amount of information is compressed into fixed-size vectors. This is known as over-squashing @alon_bottleneck_2021.
We then looked through literature in the field to find various
papers that propose solutions to alleviate over-squashing.

In our search, we found a topological approach and a geometric approach. We then came up with our own approach inspired by @alon_bottleneck_2021's approach, but with a focus on being more computationally efficient and avoiding graph rewiring.

== HubGCN (Hub aggregated Graph Convolutional Network)
We suspect that over-squashing is caused by the fact that long range interactions are being 
watered out by the neighbours around the path of information flow. 
We therefore propose adding a global node / hub to collect all the non-local information
from the graph. This will make so that the nodes have to send less non-local information
around the graph, resulting in less information that might be lost due to
over-squashing.

We do this by building on top of the SAGE framework @hamiltonYL17 and we simply add a hub node with its own set of weights (in this case, an arbitrary learnable function):
$
x^(l+1)_i &= w^l_1 + w^l_2 dot "AGG"({x^l_j: j in cal(N)(i)})+f^l (x^l_i, x^l_H)+b^l\
x^(l+1)_H &= "AGG"({x^l_j: j in V^l})
$
In this setup, we have two trainable weights, one for the local aggregation, and one for the self loop. Furthermore we have a trainable bias. Lastly there is $f^l$, this is some trainable differentiable function, in our case we let it be a two-layer MLP with a hidden dimension of 512 and `tanh` activation. The method of aggregation used in our experiments was mean aggregation, but any aggregator could be used.
The remaining ordinary GCN layers were set to have 128 hidden dimensions.

The hub node is equivalent to performing a global aggregation of the graph.
We can then compute a weighting for information from the hub node to the local nodes which is then added to each of the local nodes.

=== Experiment
To make our experiment as reproducible as possible, we chose to use Bayesian hyper-parameter optimisation (BO), first randomly initialising with 5 points in our hyper-parameter space, and then we searched using BO for an additional 15 new sets of parameters. Each parameter was then trained on once, and then the validation loss was used to select the best set. The validation loss was used as our objective, and we added a tiny penalty of $4 dot 10^(-7) dot #`epochs`)$, the motivation behind this additional penalty term was merely to reduce training times since very similar performance but significantly fewer epochs would be a preferred solution in this optimisation problem.
We later motivate our choice of using Bayesian optimisation in @ml-reproducibility.

The parameters that were searched across were:
$log_10 (#`epochs`) in [2, 3.5[$, $log_10 (#`learning_rate`) in [-4, -1[$ and $log_10 (#`C`) in [-4, -2 [$.
`C` was used as the `weight_decay` argument in the PyTorch implementation of Adam, this is equivalent to applying L2 regularisation.


We performed this search across hidden layer depths 1 through 3 and with our HubGCN method on the last layer and a baseline with a normal GCN on the last layer.
Each of these configurations was then trained on and evaluated 15 or more times#footnote[Some parameter configurations were noisy and needed more evaluations to reduce statistical error.].


=== Results
We compared the two methods with 1 to 3 hidden layers, 1 hidden layer performed better across the board and we thus only plot that for the kernel density plots, the Q-Q plot contains all three layer depths.
The observations were truncated between the 25 to 100 quantile to remove subpar runs.


// TODO: MENTION BANDWIDTH
#figure(image("assets/combined_plot.svg", width: 90%),
caption: [Comparison between HubGCN and base models.]
)

We chose to report the results using Q-Q plots and density plots. Summary statistics like the average, and standard deviation do not provide a lot of useful information when we are working with distributions that are non-gaussian, such as the resulting distributions from our experiments. Model A might outperform model B on average, but model B might rarely be better than any observation from A. This distinction might be relevant with the CiteSeer experiment where it is not immediately clear which model performs better.

The HubGCN performed significantly worse for Cora, mostly similar but slightly worse on average for CiteSeer, and significantly better for PubMed. On CiteSeer it can be noted that the single best performing observation is using HubGCN, in conclusion, more runs must be performed to rule out that HubGCN could be better but harder to train. 


== Stochastic Discrete Ricci Flow (SDRF)
The authors of @topping_understanding_2022 propose a method for improving graph representation learning via curvature based rewiring. With this they also hope to mitigate the effects of over-squashing.

Their method can be summarised as a stochastic sampling of edges and subsequent sequential mutations of the graph are performed until convergence is reached.

To gain more insights into over-squashing, we want to reproduce the results of this paper and then build on top of the results. 
This is where we hit various hurdles with reproducing the results.
The authors provide the reader with the relevant code used for the experiment on GitHub at #link("https://github.com/jctops/understanding-oversquashing")[https://github.com/jctops/understanding-oversquashing]. The code is undocumented and does not closely follow the mathematical abstractions used in the paper. The implementation is a seemingly efficient implementation relying on some undescribed linear algebra optimisations. We were unable to show that the code implemented in Python was correct and equivalent to algorithm presented in the paper. They also use a dense representation of adjacency matrices instead of a sparse graph representation which might imply that this implementations wouldn't scale to large graphs. // TODO: remove this claim?

We then began writing up our own implementation closely following the paper, which meant we had to first compute the Ricci curvature between two edges.

=== The naive implementation
To compute the Ricci curvature, we must compute three non-trivial values for a given edge $i~j$. The number of 3-cycles, the number of 4-cycles without diagonals, and the maximum number of times a specific node is visited when traversing all 4-cycles from i to j ($gamma_max$).

We implemented this entirely using sets, this is a more naive approach without any optimisation. 
For the set of nodes contained in 3-cycles ($\#_triangle (i, j)$) we noticed that this was equivalent to finding a mutual neighbour of i and j. 
// This was implemented as a set union 

The set of nodes contained in 4-cycles ($\#_square (i, j)$) were more complicated to compute. 
Bad nodes for the edge $i~j$ are defined as ${i, j} union \#_triangle (i,j)$.
We start off by finding the neighbours of $i$, then remove the bad nodes. These will be our first set of candidate nodes.
For each of these candidate nodes, we then found their neighbours, removed the bad nodes again, and then added the nodes that yielded a path to $j$.
The $gamma_max$, was found by first computing the 4-cycles on $i~j$ and counting how many times a neighbouring node was included in the cycle. This was then also done for $j~i$ to find the maximum since this approach is asymmetrical.

We chose to not explore this implementation further due to computational constraints and moved on to trying to replicate the results using the author's optimised implementation.

=== Code replication
We tried replicating the algorithm the authors provided on GitHub alongside the paper. 
Here we do not consider the correctness of the algorithm, and merely sought to replicate the algorithm and see if their implementation was significantly more efficient than our naive implementation.

We made one difference which was to use a sparse representation of our matrices to ensure that large datasets could be put into memory (datasets like PubMed would not fit into memory on most machines).

This implementation took $approx 153.0947$ seconds to compute the Balanced Forman Curvature on all edges of PubMed (88648 edges) with a speed of $approx 579.04$ edges a second. This was done on a Ryzen 5 2600 and the algorithm was re-implemented in Julia.

=== Runtime analysis
The biggest bottleneck in the SDRF algorithm is computing the Ricci / Balanced Forman Curvature (BFC) @topping_understanding_2022[p. 4]:
$
"Ric"(i, j) &:= 2/d_i + 2/d_j - 2 + 2 (|\#_triangle (i, j)|)/(max{d_i, d_j})
+ 2 (|\#_triangle (i, j)|)/(min{d_i, d_j})\
&+ 2 (gamma_max^(-1) (i, j))/(max{d_i, d_j})(|\#_square (i, j)|+|\#_square (j, i)|)
$


Let $B_r(i)$ be the set of nodes reachable from $i$ within $r$ hops.
We need to recompute the BFC for all edges in $B_1(i) union B_1(j)$ every time we add or remove an edge $i ~ j$. The number of triangles is synonymous to the number of mutual neighbours and therefore cannot be affected by neighbours of neighbours. 4-cycles are a little less intuitive but follow the same reasoning, since a 4-cycle only contains nodes reachable in $B_1(i) union B_1(j)$. Our naive set implementation only searches from outwards from $i$ and therefore needs to search around $B_2(i)$ to find $\#_square (i,j)$ which in most cases is going to be a larger set of nodes and thus less computationally efficient.

When computing the improvement vector $x$ for a given edge, we need to compute the BFC for all edges possible between $B_1(i) times B_1(j)$. This can easily become quite a large number of computations if either of the nodes happen to have a large number of neighbours.

This algorithm quickly becomes infeasible for dense graphs or large graphs due to cache utilisation. Our implementation has no efficient way of storing graphs so that neighbouring nodes have any guarantee to be in cache alongside other neighbours. This is not a trivial problem to solve since most graphs have no natural ordering of nodes or edges. The computation thus becomes very inefficient for the computations of 4-cycles and $gamma_max$. One idea could be to order the edges in the graph by the node degrees so the most connected nodes are clustered together in memory.

Furthermore, for a lot of graph datasets, the cardinality of the receptive field grows exponentially (for PubMed, the average degree is $approx 4.469$ and the maximum degree is 171). This means the algorithm risks having a very high number of nodes that need to be searched for each single computation of the curvature. This is further accentuated in datasets where there are a few nodes with a large number of neighbours, since you are very likely to hit one of these nodes and then having to search through all of the neighbours.
If we take PubMed for instance, the average degree is $approx 4.469$ and the maximum degree is $171$. The average number of paths of length 2 is $approx 75.43$ meaning the optimal implementation would on average search through $approx 150.87$ nodes for each computation of BFC which it would on average do $approx 75.43$ times.

Our implementation is much worse due to requiring paths of length 3, resulting in $approx 1191.11$ nodes being searched for each curvature computation. This happens due to the exponential growth in the number of reachable nodes.
With this insight, we re-implemented the 4-cycle search on our naive solution, making it possible to do the computation in $approx 40.04$ seconds (faster than the replication of @topping_understanding_2022 in Julia)
This improved approach on sets was not tested for correctness and was merely to see if the faster 4-cycle computation yielded tractable results.

For our naive approach, we found marginal improvements to computational performance by caching previously computed curvature values for each edge in the graph and then re-updating them every time an edge is added or removed.
This is a significant optimisation on the original implementation by @topping_understanding_2022 which recomputes the curvature across the entire graph for each iteration.

All of these improvement doesn't tackle the more fundamental problem of the algorithm essentially having to do a large number of random accesses across the graph.

The authors of @giraldo_trade-off_2023 also report not being able to reproduce the model performance of SDRF on various datasets, also due to BFC being too expensive to compute efficiently. They then substitute this metric with their own Jost-Liu Curvature (JLC) which yields worse results (comparable to the baseline GCN) than the SOTA results that @topping_understanding_2022 reported. The GitHub repository as of December 22, 2024 also has an open issue at #link("https://github.com/jctops/understanding-oversquashing/issues/3")[https://github.com/jctops/understanding-oversquashing/issues/3] from July 15, 2022 where multiple users report not being able to reproduce the same positive results from the paper. 

We chose to not investigate this algorithm further due to the aforementioned issues.


== TOGL (Topological Graph Layer)
// Performance wise we notice that the Persistance diagram computation runs the same, when single threaded, on both a low end cpu with just 3MB og cache and a midrange cpu core with 16MB cache. 
// This confirms our suspicion that the random ordering of nodes in the filtration process, is maximally not spacially local. This is because of the uniform sampling of the nodes across the datastructure which stores the graph. 
// Although the filtration could in principle be computed with high effeciency, in practice, the computation is done from the very slow system RAM and only sequencially, since the algorithm requires information from the previous step to compute the next step. This makes it inherently disadvantagous to algorithm with high spacial locality and algorithms relying on vectorizable operations.

We intuit that if a node is unable to gain information from far away nodes, as is the case for over-squashed ones, we may be able to alleviate this issue by deliberately embedding higher order graph features into the node representations, which the node would not be able to access otherwise. Specifically we will attempt to embed topological features. Topological Deep Learning (TDL) is gaining traction as a novel approach @papamarkou_position:_2024 for Graph Representation Learning (GRL). Leveraging topological higher order features, TDL has already shown promising results in alleviating various limitations of Graph Neural Networks (GNNs) @horn_topological_2022. 

Specifically, we will utilise the TOGL framework as delineated in the paper @horn_topological_2022 as a foundational reference for our topological embedding experiments, because this framework is trainable end-to-end and uses graph filtrations and persistent homology to extract the topological features. Hence in theory, we do not have to specify or select the specific features to extract. Due to the unavailability of runnable code resulting from missing dependencies and documentation, our initial objective will be to re-implement the methodology presented in the original paper, thereby contributing to the discourse on the reproducibility of machine learning research. Subsequently, we will use the tree-neighbors dataset to investigate the application of the TOGL layer. //Our aim is to assess the impact of the TOGL framework on non-standard benchmarks, specifically focusing on topologically rich datasets. We hypothesize that the TOGL layer may leverage the inherent topological structures within these datasets, potentially yielding more significant results than those reported in the original publication. This exploration will not only enhance our understanding of the TOGL framework but also contribute to the broader conversation surrounding the integration of topological methods in machine learning.

=== TOGL Layer Replication

==== The filtration

A filtration process is used to progressively "build" the graph by adding nodes one by one and checking the topological features at each step. This is done to track which topological features are persistent across process of adding nodes.The filtration function is intended to choose the order of nodes to add, based on their features. The filtration function is trainable, so that the network may presumably choose an informative node order.

The filtration process is described quite well in the original paper, in fact, we can with only small modifications replicate the exact filtration process. We simply use a feed-forward neural network with an input layer of `n_features`, a hidden layer of 32 ReLU-activated neurons, and an output layer of `n_filtrations`. We use batching to compute all n filtration in parallel for all nodes in the graph. We then sort the output for each node in each filtration, and use the indexes of the sorted nodes to represent the node order in each individual filtration. The implementation is quite straightforward and efficient since we are simply using the linear layer implementation available in `PyTorch`.

==== Dim-0 Persistence Diagram

"dim-0 features" essentially refers to the number of connected components in a graph. 

In the original paper, the theoretical underpinnings of this computation are detailed, but implementation specifics are sparse. Our approach focuses on optimizing the computational efficiency using a tailored union-find algorithm. 

We make a variant of the union-find data structure where the rank of each node corresponds to its position in the filtration order. This design ensures that components are merged according to their filtration sequence. The use of path compression optimizes the runtime complexity, minimizing the overhead associated with repeated parent lookups. It should be noted though that because of the trained filtration function, we cannot optimize for spatial or temporal locality. For example if the filtration samples uniformly across the graph, the locality of our lookup will be maximally inefficient. 

We go node by node according to the filtration order, and when we add a node which has an edge to an already existing node, the node is set to have the original node as its parent hence become part of it's connected component. We always make sure the new component is the one being set as the child, hence we will keep track of which node is the first and hence the parents of any given connected component at any time in the filtration. We keep track of the lifetime of each component. When did it appear and when did it get absorbed; we store that in a list of tuples which corresponds to our persistent homology diagram. This list of tuples is bijective with our nodes, and we can later supply each node with the information in its corresponding dim-0 persistence tuple.

==== Dim-1 Persistence Diagram

The original paper provides no description, either theoretical or implementation-related, regarding the dim-1 persistence diagram calculation. We assume that the authors are utilizing the same implementation employed in @hofer2021graphfiltrationlearning; however, this is not explicitly stated. We couldn't easily find a description in this paper either, and adding an additional paper to our project quickly got out of scope. 

==== The embedding layer
The original paper provides limited details regarding the embedding. We have chosen to utilise the DeepSets approach, as it appears to be the primary method referenced. While the authors indicate that they employ a method based on DeepSets, they do not specify which particular method is used. Consequently, we have opted to use the approach from @buterez2022graphneuralnetworksadaptive:
$ rho(sum_(i = 1)^n phi.alt_i) quad "where " phi.alt_i = phi.alt(Theta_i) $

Where both $rho$ and $phi$ are MLPs.

=== Experiment

Since we could not replicate the dim-1 embedding, we decided that it would not make sense to benchmark our implementation as is. It is still provided in the code-base for this report, but for the testing we will be using a modified version of the repo https://github.com/aidos-lab/pytorch-topological with reduced and update dependency specifications, also included in the code-base for this report. This alternative repo has an implementation of the TOGL layer written by one of the original authors in @horn_topological_2022.

There are two points of interest in this experiment, first is model accuracy; do we alleviate some kind of issue relating to over-squashing, and hence achieve better performance on our over-squashing benchmark. Secondly is performance; is the TOGL layer competitive with other GNN's.

For the first question, when tested on the tree-neighbors benchmark, the answer is unequivocally no. We fail to train most of the time already on tree depth 3. This is not entirely surprising though. For the TOGL layer to embed any kind of useful information, it must know which leaf class is the correct one for that particular graph. The graph structure itself cannot be used as all the graphs share the same structure, and as all the leaves are symmetrical (in a geometric view they are all identical, they all share the same path structure up to the root node). The problem lies in the way the filtration function works. It is applied to each node individually, and hence cannot measure interactions between nodes. We might want to test if the number of leaves of two nodes is the same (to determine the correct class of the root node), but the filtration will look at each node by itself, hence there is no opportunity for interesting behavior such as "building" the graph up from the correct node to the root, and using the embedding function to "tell" the GNN that connected components which appear early in the filtration and stick around are the correct path up to the root.

As for performance, this is more contentious. We see an enormous 2 orders of magnitude performance degradation between running with just the GCN and the GCN with the TOGL layer as can be seen in @togl-runtime-comparison. 

// this needs details of what the actual system or remove the plot and experiment
#figure(
  table(
    columns: 4,
    [], [w/o TOGL], [w/ TOGL], [slowdown factor],
    [depth], [], [], [],
    [1], [0.034759s], [3.766667s], [108.363677],
    [2], [0.040858s], [5.546667s], [135.756133],
    [3], [0.068606s], [7.636667s], [111.311691],
  ),
  caption: [runtimes given in seconds of 1 epoch with 1000 trees.]
) <togl-runtime-comparison>

Hence why we have only a few data-points in this experiment to extrapolate from. 

We suggest though that persistent homology, although flexible, is inherently slow as we must do a topological-feature calculation for each node in the graph, at each step in the filtration process. This is a sequential process where at any step $n$ the calculation depends on information from the previous steps, and additionally since it is also dependant on the node features and the graph structure, in practice, this calculation must be repeated for each new model input. With no way of vectorizing or caching the graph filtration and persistence diagrams, the TOGL layer is just very slow compared to other highly parallel algorithms which dominate the current computational paradigm of ML.  

// TODO: sub-conclusion of this section but not the whole paper
== Discussion (takeaways from experiments)

Where do we go from here? TOGL, FA and SDRF all propose a solution but
not where to continue researching from there.


// TODO: find name for this
= A framework for more reproducible benchmarks and experiments 

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

In @topping_understanding_2022, the authors propose a new algorithm for graph representation learning, 
but the naive approach (if you just directly convert the maths to code) is intractable and 
it is not feasible to train a model with this approach.
The authors provide their own code in the form of a GitHub repository.
This code might work, but we failed to get the dependencies to resolve and have the code run.
Their specific implementation uses some efficient linear algebra tricks that are not
documented in the code nor the paper, which means its not immediately apparent
that this code even is correct according to the mathematical motivation in the paper.

A lot of code has the characteristics of being swiftly prototyped code, with 
little to no documentation, no tests and no focus on portability (explicitly versioned dependencies or containerisation).
Often the code is also not modular which makes it hard to reuse the code in other 
reproductions or experiments, and you end up having to rewrite the code from scratch.


// TODO: this is repeated in the following section partially?
A way to mitigate these problems, is to better document implementation details that
deviate from the mathematics, and to provide model weights, and datasets. 
If you have the capacity to train on dozens of GPUs then you can also provide
the datasets and model weights in a portable and reproducible format.

== What is reproducibility?
We take inspiration from @Gundersen_Kjensmo_2018 and their three tiers of reproducibility. In their paper, they focus on the various tiers of reproducibility and the trade-off between concrete and more reproducible experiments vs more general methods that might be less reproducible but provide more general insights.

We are going to focus mostly on "experiment reproducible". An ML experiment is experiment reproducible when the same implementation of a method yields the same results on the same data @Gundersen_Kjensmo_2018. 
We further differentiate between the reproducibility of training, and the producibility of the model inference. 
This distinction is important since evaluating a pretrained inference method is a much simpler task than trying to reproduce the entire method pipeline. We are going to refer to these two degrees of reproducibility as "implementation reproducible" and "inference reproducible".
Concretely, inference reproducible could be implemented as in the author providing the trained model weights and the associated inference algorithm.

Another section of reproducibility that we deem important which wasn't covered by @Gundersen_Kjensmo_2018, is the same method producing similar or predictable results on different data. This is quite important for future authors trying to reproduce an experiment, since it gives them the option of evaluating the method with other metrics or on different datasets.

A concern with implementation reproducibility that we encountered more than once, was that we could reproduce a specific implementation that the authors provided in the form of code, but we couldn't show that this implementation was equivalent to the method provided in the paper which is typically a more abstract mathematical method omitting certain implementation details or essential optimisations. The authors of @giraldo_trade-off_2023 also encountered a similar set of challenges whilst trying to reproduce the results of @topping_understanding_2022 due to the computational complexity of the SDRF algorithm. 

It is therefore important to document the algorithmic optimisations that were
used to arrive at the results, be it caching, algebraic simplifications, or
other tricks. Ensuring that the path from the abstract motivation behind the method, to a concrete implementation is clear and obvious helps alleviate potential challenges in reproducing the whole implementation itself.

This doesn't tackle the problem regarding inference reproducibility, which we also encountered. This level of reproducibility serves as a sanity check and as a proof of existence of a specific method's performance.
A low hanging fruit, which we believe everyone in the field should be doing, is reporting the model weights alongside a published paper. If you have the resources to train on dozens of GPUs with terabytes of data, then it is reasonable to assume that the model weights can be stored somewhere accessible to other researchers.
Having access to the model weights, reduces the time required to evaluate a model (for instance if you need to get some unreported metrics or to compare on some other data) dramatically. It also removes any doubt that a reproduction of the inference or learning algorithm is implemented incorrectly.
We (and other researchers) encountered several times authors claiming state of the art performance using some novel approach, which we failed to reproduce. Since there are too many degrees of freedom in how the model can be implemented, trained and evaluated, the reproductions are inconclusive instead of being an unambiguous process of importing weights, and evaluating the model.



== Dependency hell and reproducibility woes in Python
Python has notoriously bad package management, and the common workflow in an academic machine learning setting, often includes the usage of either a global installation, virtual environments, or notebooks such as Jupyter or Google Colab. 
Having a single global installation of python with packages used a wide range of projects, creates major problems both in making something portable/reproducible, and ensuring no dependencies have broken. This is the problem virtual environments seek to solve, but there are multiple
standards for virtual environments, all of which are not compatible with each other and often people are met with challenges when setting them up on their systems.

When building an application that will see real-world usage, setting up version control and various processes such as code review, linting and tests, is essential. These processes are not compatible with the often-used notebook flow. The notebook files themselves are saved as JSON which Git (the most popular version control system), is incapable of effectively dealing with, making it much more difficult to track specific changes and causing the codebase to rapidly grow in size.
Notebooks also force the user into writing all the code in a linear fashion, which often clashes with how code typically branches out into isolated components and modules.

For package management you can use things like Pip, Poetry, Pipenv, Conda, uv and more. Most of these solutions are not compatible with each other. 
Some packages also cause trouble with reproducibility, for instance NumPy or other packages that depend on C++ bindings which is the culprit behind a lot of "it works on my machine" problems, where it is not immediately apparent why something should be or should not be working. These externally linked libraries are often not explicit static dependencies and might break on containerised systems or non-standard operating systems such as NixOS or systems without preinstalled C++ libraries at expected predetermined paths.
Lastly another problem with Python's ecosystem is that packages quickly become abandoned and break
due to external factors without any changes having been done to the package itself.

// TODO: mby move this? idk
A concrete example could be the experiment performed by Alon and Yahav which both uses an old
version of NumPy (there has since been made breaking changes to NumPy's API), TensorFlow, and PyTorch all
in the same experiment. There are also very few comments in the repository.
Without the high level implementation details, it is quite difficult to reproduce the results or assess the correctness of the original implementation.

// TODO MUSTAFA: pls write your legendary saga of trying to get TOGL to work with the original code

Outdated dependencies / breaking changes
Dead dependencies / Missing dependencies
No documentation or poor documentation


== Orthogonality and hyper-parameter search <ml-reproducibility>
To make our experiments more reproducible, we took inspiration from Andrew Ng and his concept of orthogonalisation. The idea is to isolate the various tunable hyper-parameters so that the hyper-parameters become easier to interpret and to optimise.

This in theory makes it possible to optimise for better models, but in our case we are interested in the orthogonality because of interpretability and the potential improvements to reproducibility. 

In machine learning, we seek to minimise the bias and variance of a given model applied to some data. In most cases, we will be tuning some hyper-parameter that positively affects the bias and negatively affects the variance or vice versa.
This can be further broken down to various steps:
1. Train the model to perform well on the training data.
2. Evaluate the model and perform model selection optimising on validation data.
3. Evaluate the model on test and ensure the test metrics are aligned with actual performance on "real" data.

//This can be broken down to two separate problems, minimising how much the model overfits, and minimising the objective function (typically loss). 

To address model performance on training data, we can improve the size of the dataset, or improve the model either by making it more well-suited to the data or by increasing complexity.
After this, we might choose to select the best performing model either given various configurations of hyperparameters or various architectures. 
At this step, we might also choose to use regularisation to restrict the model and make it less likely to overfit.

This leads into our motivation for using Bayesian optimisation in place of early-stopping in the latter half of the paper. Early-stopping is a popular choice in machine learning where the model trains until it stops improving, then the best model observed is selected.
This has the effect of improving performance on the train data by changing the number of epochs as well as improving the performance on the validation data by alleviating overfitting. This affects multiple phases of the training process at once. 
We therefore suggest tuning the number of epochs as a hyperparameter alongside some method of model regularisation such as L2.
Another potential problem is performing a grid search for hyperparameters, these are often done with arbitrarily chosen values and they have no guarantees on good coverage of the parameter space. We therefore also use Bayesian optimisation as it allows us to specify a more broad and less arbitrary range of infinitely many values that could be selected.  

Bayesian optimisation itself also has some parameters that need to either be chosen ahead of time, or by optimisation. In our experiments, we used the default settings of the BayesianOptimisation package in Python. 
Selecting a different acquisition function or kernel can yield different results. Model evaluations are often quite noisy, and will potentially require an adjustment to the level of noise expected by the chosen kernel. Alternatively multiple observations could be done for a given set of hyperparameters, but this is significantly more expensive from a computational point of view. 
// TODO: might need to elaborate why
We chose not to optimise the parameters of our search. 


// TODO:
// L2, dropout and early stopping are equivalent, find source to why

// chain of assumptions (andrew ng)
// Fit on train and perform well
// Fit on dev and perform well
// Fit on test and perform well
// This implies that it performs well in the real world (generalises)
// https://thevivekpandey.github.io/posts/2017-10-22-deeplearning-coursera-course-3.html
// https://cs230.stanford.edu/files/C3M1.pdf

== Metrics and reporting of results
Most papers (source) report the results as either the best result or the average across
multiple runs with a standard deviation. These two summary statistics in isolation are
not as informative as plots on the different distributions of models.

We therefore suggest plot types such as kernel density, Q-Q or candlestick plots.
Typically, we are interested in producing the model that maximises some performance metric as a result of some training and selection process.
Equivalently, this is the maximum value drawn from some sample of model evaluations.
But there is a catch, the maximum of a sample of models from some distribution is a statistically meaningless number which cannot be compared to other distributions.
It tells us virtually nothing about the distribution other than the maximum value existing in the support.

The average is more meaningful but doesn't help us with comparing two distributions if we are looking for the distribution that gives us the best performing models overall. It is only helpful if we know the distributions exhibit similar behaviour (for instance, if they are both gaussian with equal variance).

It is also important to ensure that the estimates are statistically significant and have a low uncertainty, this could be by sampling more data until the standard error is at an acceptable level. 
Without these constraints on the distributions, a distribution with a lower average might still have a tail that contains better models.

Going back to inference reproducibility, we also note another important benefit of reporting the model weights alongside the paper and code. Publishing the weights also makes it possible for authors to quantitatively compare models from several unrelated papers that may or may not be using different metrics, some use unbiased losses, some use losses, some might use an entirely different metric and some might be using various aggregations of these metrics.


// TODO: ANOTHER NOTE (yes again shhh): what about cross validation we dont mention that

// They done done it Don, then they did damn did it didnt they?

#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])
#show heading: it => {
  if it.level == 1 and it.numbering != none {
    [#it.supplement #counter(heading).display():]
  } else if it.numbering != none {
    [#counter(heading).display().]
  }

  h(0.3em)
  it.body
  parbreak()
}

= Class distributions for three-node classification <appendix-1>
Let $X$ represent the sum of two independently sampled discrete distributions, from 0 to 9. The possible values of $X$ range from 0 ($0+0$) to 18 ($9+9$). The probability distribution of $X$ forms a triangular distribution.

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
// TODO LIST
// tree-neighbors
// [ ] 4e. Write discussion for tree neighbours - Josh/Mustafa

// [ ] 5. TOGL - Mustafa
// [ ] 7. Write discussion for all experiments
// [ ] 8. ML Framework - To be assigned
// [ ] 11. replot TODO: placeholder plot 1 in julia (???)
// [ ] 12. make doc and plots use same fonts (svg)
// [ ] 13. Choice of template (line spacing, numbering etc) - Josh/Mustafa
// [ ] 14. fix package management in julia
// TODO: Link to our github
// TODO: figure out where we store model weights
// TODO: clean up the curvature.py file, it has hardcoded values and is ugly

// TODO: Some places we use the XOR thing to represent AGG and other places we use AGG, why?