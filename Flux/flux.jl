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

# Download the data, and create traint and test samples
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

# Compute the accuracy error to minimize
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
    
    # get suggested hyperparameters
    @suggest activation in trial
    @suggest n_dense in trial
    @suggest dense in trial

    # Create the model with dense layers (fully connected)
    ann = []
    n_input = prod(imgsize)
    for n in dense[1:n_dense]
        push!(ann, Dense(n_input, n, activation))
        n_input = n
    end
    push!(ann, Dense(n_input, nclasses))
    model = Chain(ann) |> device
    # model parameters
    ps = Flux.params(model)  

    # hyperparameters for the optimizer
    @suggest η in trial
    @suggest λ in trial

    # Instantiate the optimizer
    opt = λ > 0 ? Flux.Optimiser(WeightDecay(λ), ADAM(η)) : ADAM(η)

    # get suggested loss
    @suggest loss in trial
    accuracy_error = 1.0

    epochs = 20 # maximum number of training epochs
    # Training
    for epoch in 1:epochs
        for (x, y) in train_loader
            # batch computation
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end
        # Compute intermediate accuracy error
        accuracy_error = eval_accuracy_error(test_loader, model, device)
        # report value to pruner
        report_value!(trial, accuracy_error)
        # check if pruning is necessary
        should_prune(trial) && (return)
    end
    
    # if accuracy is over 90%, then trials is considered as feasible
    accuracy_error < 0.1 && report_success!(trial)
    # return objective function value
    accuracy_error
end

# maximum and minimum number of dense layers
const MIN_DENSE = 2
const MAX_DENSE = 5

scenario = Scenario(### hyperparameters
                    # learning rates
                    η = (0.0..0.5),
                    λ = (0.0..0.5),
                    # activation functions
                    activation = [leakyrelu, relu],
                    # loss functions
                    loss = [mse, logitcrossentropy],
                    # number of dense layers
                    n_dense = MIN_DENSE:MAX_DENSE,
                    # number of neurons for each dense layer
                    dense   = Bounds(fill(4, MAX_DENSE), fill(128, MAX_DENSE)),
                    ### Common settings
                    pruner= MedianPruner(start_after = 5#=trials=#, prune_after = 10#=epochs=#),
                    verbose = true, # show the log
                    max_trials = 30, # maximum number of hyperparameters computed
                   )

display(scenario)

# minimize accuracy error
HyperTuning.optimize(objective, scenario)

