@recipe function plot_recipe(sim::Simulation)

    layout = zeros(Int,1,0) 
    sim.mpc.model.ny == 0 || (layout = [layout (sim.mpc.model.ny, 1)])
    sim.mpc.model.nu == 0 || (layout = [layout (sim.mpc.model.nu, 1)])
    layout := layout

    # Plot y
    id = 0
    for i in 1:sim.mpc.model.ny
        @series begin
            i == ny && (xguide --> "Time (s)")
            yguide  --> sim.mpc.model.labels.y[i]
            color   --> 1
            subplot --> id 
            #label   --> "\$\\mathbf{y}\$"
            legend  --> false
            sim.ts, sim.ys[i, :]
        end
        id+=1
    end
    # --- manipulated inputs u ---
    for i in 1:nu
        i_u = indices_u[i]
        @series begin
            i == nu && (xguide --> "Time (s)")
            yguide     --> sim.mpc.model.labels.u[i]
            color      --> 1
            subplot    --> i+=1
            seriestype --> :steppost
            #label      --> "\$\\mathbf{u}\$"
            legend     --> false
            sim.ts, sim.us[i_u, :]
        end
    end

    ## --- plant states x ---
    #for i in 1:nx
    #    i_x = indices_x[i]
    #    @series begin
    #        i == nx && (xguide --> "Time (s)")
    #        yguide     --> xname[i_x]
    #        color      --> 1
    #        subplot    --> subplot_base + i
    #        label      --> "\$\\mathbf{x}\$"
    #        legend     --> false
    #        t, res.X_data[i_x, :]
    #    end
    #end
end
