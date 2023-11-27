using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays

x, y = Iris(as_df=false)[:];
x = Float32.(x);
y = vec(y)

custom_y_onehot = unique(y) .== permutedims(y)

const classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
flux_y_onehot = onehotbatch(y, classes)

m(W, b, x) = W * x .+ b

W = rand(Float32, 3, 4);
b = [0.0f0, 0.0f0, 0.0f0];