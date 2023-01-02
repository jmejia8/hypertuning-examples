using MLJ, MLJLinearModels
using HyperTuning

# data
const iris = load_iris()
const y, X = unpack(iris, ==(:target); rng=2023)

# classifier error for hyperparameters λ, γ
function objective(trial)
    @suggest lambda in trial
    @suggest gamma in trial
    @suggest classifier in trial
    @suggest penalty in trial
    @suggest fit_intercept in trial
    method = classifier(;lambda, gamma, penalty, fit_intercept)


    eval_res = nothing
    try
        eval_res = evaluate(
                            method, X, y,
                            resampling=CV(shuffle=true), # cross validation
                            measures=[accuracy],
                            verbosity=0,
                           )
    catch e
        id = trial.value_id
        @error "Trial $id errored."
        println(e)
        return 1.0
    end

    1 - eval_res.measurement[1] # accuracy error
end


scenario = Scenario(
                    # hyperparameters
                    lambda = (0.0..1.0),
                    gamma  = (0.0..1.0),
                    classifier = [MultinomialClassifier, LogisticClassifier],
                    penalty = [:l1, :l2, :en, :none],
                    fit_intercept = [true, false],
                    # brute force 
                    verbose = true,
                    max_trials = 100,
                   )

# hyperparameter optimization
HyperTuning.optimize(objective, scenario)

@info "Top parameters"
# show the list of top parameters regarding success, accuracy, and time
display(top_parameters(scenario))

@info "History"
# display all evaluated trials
display(history(scenario))
