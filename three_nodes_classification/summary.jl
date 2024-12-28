using CSV
using DataFrames
using Plots; gr(markerstrokewidth=0)
using Plots.PlotMeasures
using StatsPlots
using Distributions
using HypothesisTests

results_folder = "results"

results = CSV.read(joinpath(results_folder, "runs.csv"), DataFrame)
results[!, :score] = results[!, :accuracy] / 100

# Filter out really bad runs since we are only interested
# in seeing if the FA method is better in the good runs
min_q = 0.25

function plot_score_distribution(df, λ=0.003)
    min_x = quantile(df.score, min_q) - 0.002
    max_x = maximum(df.score) + 0.005


    p=plot()

    density!(df.score, label=nothing, color=:blue, alpha=0.5, bandwidth=λ)

    # Plot observed data as scatter on the bottom
    y = repeat([0], nrow(df))

    scatter!(df.score, y, label=nothing, mc=:blue, alpha=0.25)

    xlims!(min_x, max_x)
    xlabel!("Accuracy")
    
    plot!(size=(400, 400))  

    p
end

# Generate and display the plot
output_plot = plot_score_distribution(results, 0.01)
savefig(output_plot, "results/kde-mlp2-2-bcelosswlogits.svg")
