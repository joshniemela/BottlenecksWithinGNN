using CSV
using DataFrames
using Plots; gr()
using StatsPlots
using Distributions
using HypothesisTests

results_folder = "results"

results = CSV.read(joinpath(results_folder, "runs.csv"), DataFrame)
cora = results[results.dataset_name .== "Cora", :]
citeseer = results[results.dataset_name .== "CiteSeer", :]
pubmed = results[results.dataset_name .== "PubMed", :]

function plot_score_distribution(df, n=1)
    w_adj = df[(df."use_fully_adj" .== 1) .& (df."num_hidden_layers" .== n), :]
    w_no_adj = df[(df."use_fully_adj" .== 0) .& (df."num_hidden_layers" .== n), :]
    # print means
    println("mean w_adj: ", mean(w_adj.score))
    println("mean w_no_adj: ", mean(w_no_adj.score))
    println("T-value:", UnequalVarianceTTest(w_adj.score, w_no_adj.score))

    bandwidth = 0.01

    density!(w_adj.score, label="With fully adj", color=:blue, alpha=0.5, bandwidth=bandwidth)
    density!(w_no_adj.score, label="Without fully adj", color=:red, alpha=0.5, bandwidth=bandwidth)
    xlabel!("Score")
end


function plot_score_qq_by_hidden_layers(df)
    bandwidth = 0.01
    hidden_layers = unique(df."num_hidden_layers")
    p=plot()  # Initialize the plot

    for h in hidden_layers
        # Filter data by hidden layers and adjustment usage
        w_adj = df[(df."num_hidden_layers" .== h) .& (df."use_fully_adj" .== 1), :]
        w_no_adj = df[(df."num_hidden_layers" .== h) .& (df."use_fully_adj" .== 0), :]

        # Add density plots for each group
        if !isempty(w_adj) && !isempty(w_no_adj)
            qqplot!(w_no_adj.score, w_adj.score, label="HL: $h", legend=:topleft)
        end
    end

    xlabel!("Base")
    ylabel!("FA")
    title!("Accuracy FA vs Base")
    # make x and y equal
    #min_xy = minimum(df.Score) + 0.05
    #max_xy = maximum(df.Score) + 0.01
    #xlims!(min_xy, max_xy)
    #ylims!(min_xy, max_xy)

    p
end

