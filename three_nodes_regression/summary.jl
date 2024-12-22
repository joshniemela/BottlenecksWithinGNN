using CSV
using DataFrames
using StatsBase

results_folder = "results"

results = CSV.read(joinpath(results_folder, "runs.csv"), DataFrame)

normalised = filter(row -> row.normalise, results)
not_normalised = filter(row -> !row.normalise, results)
