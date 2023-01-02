using HyperTuning
import HardTestProblems
import Metaheuristics

function getproblem(id)
    f, conf = HardTestProblems.get_cec2020_problem(id)
    fmin = conf[:minimum]

    return f, [conf[:xmin] conf[:xmax]], fmin
end


function objective(trial)
    # get instance information
    f, searchspace, fmin = getproblem(get_instance(trial))
    # get provided seed for the RNG
    seed = get_seed(trial)

    # update this to select another metaheuristic
    @unpack N, K, η_max = trial
    options = Metaheuristics.Options(;seed)
    metaheuristic = Metaheuristics.ECA(;N, K, η_max, options)

    # metaheuristic loop
    while !Metaheuristics.should_stop(metaheuristic)
        Metaheuristics.optimize!(f, searchspace, metaheuristic)
        # get best solution found found so far
        fmin_approx = Metaheuristics.minimum(metaheuristic.status)
        # report value to pruner
        report_value!(trial, fmin_approx)
        # check if pruning is required
        should_prune(trial) && (return)
    end

    fmin_approx = Metaheuristics.minimum(metaheuristic.status)
    # check if desired accuracy is met
    fmin_approx - fmin < 1e-8 && report_success!(trial)
    # return value obtained by the metaheuristic
    fmin_approx
end

function configure_eca()
    @info "Loading scenario..."
    scenario = Scenario(# hyperparameters:
                        N = range(20, 200, step=10), # 20, 30,..., 200
                        K = 3:15,                    # 3, 4, 5, ..., 15
                        η_max = (0.0..3.0),          # interval [0, 3]
                        # general settings
                        instances  = 1:10, # instances from 1 to 10
                        max_trials = 100,
                        verbose    = true,
                        # prune after 100 iterations of the metaheuristic
                        pruner     = MedianPruner(prune_after = 100), 
                       )

    HyperTuning.optimize(objective, scenario)
    scenario
end

scenario = configure_eca()

@info "Top parameters"
# show the list of top parameters regarding success, accuracy, and time
display(top_parameters(scenario))

@info "History"
# display all evaluated trials
display(history(scenario))

# obtain the hyperparameters values
@unpack N, K, η_max = scenario;
