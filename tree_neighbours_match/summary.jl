using CSV
using DataFrames
using Plots; gr(markerstrokewidth=0)
using Plots.PlotMeasures
using StatsPlots

results_folder = "."

results = CSV.read(joinpath(results_folder, "data.csv"), DataFrame)

# Filter out really bad runs since we are only interested
# in seeing if the FA method is better in the good runs
min_q = 0.0

function plot_score_distribution(df, n=1, λ=0.003)
    w_adj = df[(df."use_fully_adj" .== 1) .& (df."tree_depth" .== n), :]
    w_no_adj = df[(df."use_fully_adj" .== 0) .& (df."tree_depth" .== n), :]

    min_x = quantile(df.score, min_q) - 0.002
    max_x = maximum(df.score) + 0.002


    p=plot()  # Initialize the plot

    density!(w_no_adj.score, label="Base", color=:red, alpha=0.5, bandwidth=λ)
    density!(w_adj.score, label="Last-FA", color=:blue, alpha=0.5, bandwidth=λ)

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

    for (i, h) in enumerate(unique(df."tree_depth"))
        # Filter data by hidden layers and fully adjacent
        w_adj = df[(df."tree_depth" .== h) .& (df."use_fully_adj" .== 1), :]
        w_no_adj = df[(df."tree_depth" .== h) .& (df."use_fully_adj" .== 0), :]
        # normalise to w_no_adj
        w_no_adj.score = (w_no_adj.score .-2.0^-h) ./ (maximum(w_no_adj.score) .- 2.0^-h)
        w_adj.score = (w_adj.score .-2.0^-h) ./ (maximum(w_no_adj.score) .- 2.0^-h)

        # Add density plots for each group
        if !isempty(w_adj) && !isempty(w_no_adj)
            qqplot!(w_no_adj.score, w_adj.score, label="HL: $h", legend=:topleft, qqline=:none, alpha=0.9, color=i, ms=2)
        end
        hline!([maximum(w_adj.score)], label=nothing, legend=:topleft, color=i, alpha=0.5)
    end

    w_no_adj = df[(df."use_fully_adj" .== 0), :]
    w_adj = df[(df."use_fully_adj" .== 1), :]

    xlabel!("Base")
    ylabel!("Last-FA")

    min_xy = 0
    max_xy = 1

    max_x = maximum(w_no_adj.score) + 0.01
    max_y = maximum(w_adj.score) + 0.03

    # Add identity line
    plot!([min_xy, max_xy], [min_xy, max_xy], color=:black, linestyle=:dash, label=nothing)

    xlims!(min_xy, max_x)
    ylims!(min_xy, max_y)

    p
end

function plot_score_line_by_tree_depth(df)
    p = plot()

    depths = unique(df."tree_depth") |> sort
    max_scores_fa = []
    max_scores_baseline = []

    for h in depths
        # Filter data by depth
        w_adj = df[(df."tree_depth" .== h) .& (df."use_fully_adj" .== 1), :]
        w_no_adj = df[(df."tree_depth" .== h) .& (df."use_fully_adj" .== 0), :]

        push!(max_scores_fa, maximum(w_adj.score))
        push!(max_scores_baseline, maximum(w_no_adj.score))
    end

    # Add lines to the plot
    plot!(depths, max_scores_baseline, label="Base", lw=2, marker=:circle, color=:blue)
    plot!(depths, max_scores_fa, label="Last-FA", lw=2, marker=:square, color=:red)

    max_scores_mlp = [1.0, 1.0, 0.997, 0.3908]
    plot!(depths, max_scores_mlp, label="MLP-agg", lw=2, marker=:square, color=:green)

    xlabel!("Tree Depth")
    ylabel!("Max Accuracy")

    p
end

line_plot = plot_score_line_by_tree_depth(results)
savefig(line_plot, "results/tree_neighbor_accuracy.svg")


qq_plot = plot_score_qq_by_hidden_layers(results)
savefig(qq_plot, "results/tree_neighbor_qq.svg")

data = CSV.read("data.csv", DataFrame)

function plot_score_distribution(df::DataFrame, tree_depth, λ=0.01)
    filtered_df_no_fa = filter(row -> row.tree_depth == tree_depth && row.use_fully_adj == false, df)
    filtered_df_fa = filter(row -> row.tree_depth == tree_depth && row.use_fully_adj == true, df)

     # Check if there is data for either filtered DataFrame
     if nrow(filtered_df_no_fa) == 0 && nrow(filtered_df_fa) == 0
        return plot(title="No data for tree_depth=$tree_depth")
    end

    p = plot()

    # Plot density for the first filtered DataFrame (no fully adjusted)
    if nrow(filtered_df_no_fa) > 0
        density!(filtered_df_no_fa.score, label="No FA layer", color=:blue, alpha=0.5, bandwidth=λ)
        y_no_fa = repeat([0], nrow(filtered_df_no_fa))
        scatter!(filtered_df_no_fa.score, y_no_fa, label=nothing, mc=:blue, alpha=0.25)
    end

    # Plot density for the second filtered DataFrame (fully adjusted)
    if nrow(filtered_df_fa) > 0
        density!(filtered_df_fa.score, label="w/ FA layer", color=:red, alpha=0.5, bandwidth=λ)
        y_fa = repeat([0], nrow(filtered_df_fa))
        scatter!(filtered_df_fa.score, y_fa, label=nothing, mc=:red, alpha=0.25)
    end

    # Set plot limits and labels
    xlims!(0, 1)
    xlabel!("Score")
    ylabel!("Density")
    title!("Layer Depth: $tree_depth")
    plot!(size=(400, 400))

    return p
end

tree_depths = unique(data.tree_depth)

plots = []
for tree_depth in tree_depths
    push!(plots, plot_score_distribution(data, tree_depth))
end

plot_grid = plot(plots..., layout=grid(length(tree_depths), 1), size=(500, 1000))

savefig(plot_grid, "results/kde_plots_by_tree_depth.svg")
