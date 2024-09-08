using DataFrames
using CSV
using Distributions


# Both experiments were conducted 5 times
df = CSV.read("table1-gat.csv", DataFrame)

function welch_t_statistic(x1, x2, s1, s2, n1, n2)
    Δm = x1 - x2
    s = sqrt(s1^2/n1 + s2^2/n2)

    Δm / s
end


function welch_df(s1, s2, n1, n2)
    v1 = s1^2 / n1
    v2 = s2^2 / n2

    (v1 + v2)^2 / (s1^4 / (n1^2 * (n1 - 1)) + s2^4 / (n2^2 * (n2 - 1)))
end


function welch_test(x1, x2, s1, s2, n1, n2)
    t = welch_t_statistic(x1, x2, s1, s2, n1, n2)
    df = welch_df(s1, s2, n1, n2)

    tdist = TDist(df)

    # Right-tailed test
    1 - cdf(tdist, t)
end


# Perform t-test on the data in the dataframe, we have base-mean, base-stdev, +FA-mean and +FA-stdev

# Base
base_mean = df[!, "base-mean"]
base_stdev = df[!, "base-stdev"]

# +FA
fa_mean = df[!, "+FA-mean"]
fa_stdev = df[!, "+FA-stdev"]

n = 5

p_values = [welch_test(base_mean[i], fa_mean[i], base_stdev[i], fa_stdev[i], n, n) for i in 1:size(df, 1)]

df[!, "p-value"] = p_values

# Now find the percentage differences for the means
diffs = (fa_mean .- base_mean) ./ base_mean

df[!, "diff"] = diffs

# Filter out all the not significant variables
sig_df = df[df[!, "p-value"] .< 0.01, :]
sig_df = sort(sig_df, :diff)
