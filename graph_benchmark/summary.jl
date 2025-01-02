using CSV
using DataFrames
using Plots; gr(markerstrokewidth=0)
using Plots.PlotMeasures
using StatsPlots
using Distributions
using HypothesisTests

results_folder = "results"

results = CSV.read(joinpath(results_folder, "runs.csv"), DataFrame)
cora = results[results.dataset_name .== "Cora", :]
citeseer = results[results.dataset_name .== "CiteSeer", :]
pubmed = results[results.dataset_name .== "PubMed", :]

# Filter out really bad runs since we are only interested
# in seeing if the FA method is better in the good runs
min_q = 0.25

function plot_score_distribution(df, n=1, λ=0.003)
    w_adj = df[(df."use_fully_adj" .== 1) .& (df."num_hidden_layers" .== n), :]
    w_no_adj = df[(df."use_fully_adj" .== 0) .& (df."num_hidden_layers" .== n), :]

    min_x = quantile(df.score, min_q) - 0.002
    max_x = maximum(df.score) + 0.002


    p=plot()  # Initialize the plot

    density!(w_no_adj.score, label="Base", color=:red, alpha=0.5, bandwidth=λ)
    density!(w_adj.score, label="HubGCN", color=:blue, alpha=0.5, bandwidth=λ)

    # Plot observed data as scatter on the bottom
    y_w_adj = repeat([0], nrow(w_adj))
    y_w_no_adj = repeat([0], nrow(w_no_adj))

    scatter!(w_no_adj.score, y_w_no_adj, label=nothing, mc=:red, alpha=0.5)
    scatter!(w_adj.score, y_w_adj, label=nothing, mc=:blue, alpha=0.5)

    xlims!(min_x, max_x)
    xlabel!("Accuracy")

    p
end

function plot_score_qq_by_hidden_layers(df)
    p=plot()

    for h in unique(df."num_hidden_layers")
        # Filter data by hidden layers and fully adjacent
        w_adj = df[(df."num_hidden_layers" .== h) .& (df."use_fully_adj" .== 1), :]
        w_no_adj = df[(df."num_hidden_layers" .== h) .& (df."use_fully_adj" .== 0), :]

        # Add density plots for each group
        if !isempty(w_adj) && !isempty(w_no_adj)
            qqplot!(w_no_adj.score, w_adj.score, label="HL: $h", legend=:topleft, qqline=:none)
        end
    end

    w_no_adj = df[(df."use_fully_adj" .== 0), :]
    w_adj = df[(df."use_fully_adj" .== 1), :]

    xlabel!("Base")
    ylabel!("HubGCN")

    min_xy = quantile(df.score, min_q) - 0.002
    max_xy = maximum(df.score)

    max_x = maximum(w_no_adj.score) + 0.002
    max_y = maximum(w_adj.score) + 0.002

    # Add identity line
    plot!([min_xy, max_xy], [min_xy, max_xy], color=:black, linestyle=:dash, label=nothing)

    xlims!(min_xy, max_x)
    ylims!(min_xy, max_y)

    p
end

# Visualise the results
function create_combined_plot()
    datasets = [(cora, "Cora"), (citeseer, "CiteSeer"), (pubmed, "PubMed")]

    plots = []  # Collect all plots for layout

    for (data, name) in datasets
        # Plot score distribution
        dist_plot = plot_score_distribution(data)
        plot!(dist_plot, title="HL: 1, $name")

        # Plot QQ plot
        qq_plot = plot_score_qq_by_hidden_layers(data)
        plot!(qq_plot, title="Q-Q, $name")

        # Append to the list of plots
        push!(plots, dist_plot)
        push!(plots, qq_plot)
    end

    # Create the combined layout
    combined_plot = plot(plots..., layout=(3, 2), size=(1000, 1000), left_margin=5mm)
    combined_plot
end

# Generate and display the combined plot
combined_plot = create_combined_plot()
savefig(combined_plot, "results/combined_plot.svg")
