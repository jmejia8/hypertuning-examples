using HyperTuning
import Optim
import LineSearches as LS

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
const x0 = [0.0, 0.0]

function objective(trial)
    @suggest alphaguess in trial
    @suggest linesearch in trial
    @suggest algorithm in trial
    @suggest α0 in trial

    if alphaguess <: LS.InitialPrevious || alphaguess <: LS.InitialStatic
        ag = alphaguess(;alpha = α0)
    else
        ag = alphaguess(;α0)
    end
    

    algo = algorithm(;alphaguess = ag, linesearch = linesearch)
    res = Optim.optimize(f, x0, algo, Optim.Options(g_tol = 1e-12, iterations = 10))
    Optim.minimum(res)
end



scenario = Scenario(
                    # hyperparameters
                    alphaguess = [
                                  LS.InitialPrevious,
                                  LS.InitialStatic,
                                  LS.InitialHagerZhang,
                                  LS.InitialQuadratic,
                                  LS.InitialConstantChange,
                                 ],
                    linesearch = [
                                  LS.HagerZhang(),
                                  LS.BackTracking(),
                                  LS.StrongWolfe(),
                                  LS.Static()
                                 ],
                    α0 = range(0.1, 1, step=0.1),
                    algorithm = [Optim.Newton, Optim.BFGS],
                    # brute force 
                    sampler = GridSampler(),
                    verbose = true,
                   )

# hyperparameter optimization
HyperTuning.optimize(objective, scenario)

@info "Top parameters"
# show the list of top parameters regarding success, accuracy, and time
display(top_parameters(scenario))

@info "History"
# display all evaluated trials
display(history(scenario))

@unpack alphaguess, linesearch, algorithm = scenario;
