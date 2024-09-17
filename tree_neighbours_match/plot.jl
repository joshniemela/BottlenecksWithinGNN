using Plots
using CSV
using DataFrames
using Statistics
using StatsPlots



df = CSV.read("data.csv", DataFrame)
df.fully_adjacent_last_layer = ifelse.(df.fully_adjacent_last_layer .== true, "w/ FA", "base")

@df df groupedboxplot(
    :tree_depth,
    :max_train_accuracy,
    group=:fully_adjacent_last_layer,
    xlabel="Tree depth",
    ylabel="Max train accuracy",
    legend=:topright,
    yticks=0:0.1:1
)
