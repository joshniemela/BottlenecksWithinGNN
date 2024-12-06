using CSV
using DataFrames
using Plots; gr()
using StatsPlots

results_folder = "results"

citeseer = CSV.read(joinpath(results_folder, "CiteSeer.csv"), DataFrame)
cora = CSV.read(joinpath(results_folder, "Cora.csv"), DataFrame)
pubmed = CSV.read(joinpath(results_folder, "PubMed.csv"), DataFrame)

function plot_score_distribution(df)
    w_adj = df[df."Use Fully Adj" .== 1, :]
    w_no_adj = df[df."Use Fully Adj" .== 0, :]

    bandwidth = 0.01

    density(w_adj.Score, label="With fully adj", color=:blue, alpha=0.5, bandwidth=bandwidth)
    density!(w_no_adj.Score, label="Without fully adj", color=:red, alpha=0.5, bandwidth=bandwidth)
    xlabel!("Score")
end


function plot_score_qq_by_hidden_layers(df)
    bandwidth = 0.01
    hidden_layers = unique(df."Num Hidden Layers")
    p=plot()  # Initialize the plot

    for h in hidden_layers
        # Filter data by hidden layers and adjustment usage
        w_adj = df[(df."Num Hidden Layers" .== h) .& (df."Use Fully Adj" .== 1), :]
        w_no_adj = df[(df."Num Hidden Layers" .== h) .& (df."Use Fully Adj" .== 0), :]

        # Add density plots for each group
        if !isempty(w_adj) && !isempty(w_no_adj)
            qqplot!(w_no_adj.Score, w_adj.Score, label="HL: $h", legend=:topleft)
        end
    end

    xlabel!("Base")
    ylabel!("FA")
    title!("QQ plot of accuracy FA vs Base")
    # make x and y equal
    #min_xy = minimum(df.Score) + 0.05
    #max_xy = maximum(df.Score) + 0.01
    #xlims!(min_xy, max_xy)
    #ylims!(min_xy, max_xy)

    p
end

    
