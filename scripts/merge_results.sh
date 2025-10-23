# MERGE JSON FILES INTO A SINGLE CSV
julia --project=. -e '
using JSON, Glob, DataFrames, CSV

files = glob("results/benchmark_*.json", ".")
rows = [JSON.parsefile(file) for file in files]

df = DataFrame(rows)
df = sort(df, [:M, :P, :N, :F, :T])
df = df[:, [:P, :N, :F, :M, :T]] 

CSV.write("results/benchmarks.csv", df)
println("Saved results/benchmarks.csv")
'


