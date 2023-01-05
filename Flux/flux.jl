# adapted from: https://github.com/FluxML/model-zoo


using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs, flatten
using Flux.Losses: crossentropy, logitcrossentropy, mse, kldivergence
using MLDatasets
using Base: @kwdef
using HyperTuning
import Random
using CUDA

function getdata(;batchsize = 256)

    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    imgsize=(28,28,1)
    nclasses=10

    xtrain = flatten(xtrain)
    xtest  = flatten(xtest)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=batchsize)

    return train_loader, test_loader, imgsize, nclasses
end



function eval_accuracy_error(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return 1 - acc/ntot
end


function objective(trial)
    # fix seed for the RNG
    seed = get_seed(trial)
    Random.seed!(seed)
    # activate CUDA if possible
    device = CUDA.functional() ? gpu : cpu
    # Create test and train dataloaders
    train_loader, test_loader, imgsize, nclasses = getdata()
    
    # get activation function
    @suggest activation in trial
    @suggest n_dense in trial
    @suggest dense in trial

    # dense layers (fully connected)
    ann = []
    n_input = prod(imgsize)
    for n in dense[1:n_dense]
        push!(ann, Dense(n_input, n, activation))
        n_input = n
    end

    push!(ann, Dense(n_input, nclasses))
    model = Chain(ann) |> device
    ps = Flux.params(model)  

    # hyperparameters for the optimizer
    @suggest η in trial
    @suggest λ in trial

    # Optimizer
    opt = λ > 0 ? Flux.Optimiser(WeightDecay(λ), ADAM(η)) : ADAM(η)

    @suggest loss in trial
    accuracy_error = 1.0

    epochs = 20
    # Training
    for epoch in 1:epochs
        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        accuracy_error = eval_accuracy_error(test_loader, model, device)
        report_value!(trial, accuracy_error)
        should_prune(trial) && (return)
    end

    accuracy_error < 0.1 && report_success!(trial)

    accuracy_error
end

const MIN_DENSE = 2
const MAX_DENSE = 5

scenario = Scenario(
                    activation = [leakyrelu, relu],
                    loss = [mse, logitcrossentropy],
                    η = (0.0..0.5),
                    λ = (0.0..0.5),
                    n_dense = MIN_DENSE:MAX_DENSE,
                    dense   = Bounds(fill(4, MAX_DENSE), fill(128, MAX_DENSE)),

                    pruner= MedianPruner(start_after = 5, prune_after = 10),
                    verbose = true,
                    max_trials = 30,
                   )

HyperTuning.optimize(objective, scenario)
