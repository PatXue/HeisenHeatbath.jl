module HeisenHeatbath

using Carlo
using HDF5

export HeisenHeatbathMC

# Note: Using temperature in units of energy (k_B = 1)
mutable struct HeisenHeatbathMC <: AbstractMC
    T::Float64 # Temperature
    J::Float64 # Interaction energy
    H::Float64 # External field
    spins::Matrix{NTuple{3, Float64}}
end

function HeisenHeatbathMC(params::AbstractDict)
    Lx, Ly = params[:Lx], params[:Ly]
    T = params[:T]
    J = params[:J]
    H = params[:H]
    return HeisenHeatbathMC(T, J, H, fill((0, 0, 0), Lx, Ly))
end

function Carlo.init!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext, params::AbstractDict)
    if params[:rand_init]
        mc.spins .= rand(ctx.rng, (-1, 1), size(mc.spins))
    else
        mc.spins .= 1
    end
    return nothing
end

function Carlo.sweep!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext)
    # Select site to propose spin flip
    Lx, Ly = size(mc.spins)
    for _ in 1:length(mc.spins)
        x = rand(ctx.rng, 1:Lx)
        y = rand(ctx.rng, 1:Ly)

        # Sum of nearest neighbors' spins
        adj_spin_sum = mc.spins[mod1(x-1, Lx), y] + mc.spins[x, mod1(y-1, Ly)] +
                       mc.spins[mod1(x+1, Lx), y] + mc.spins[x, mod1(y+1, Ly)]
        # Energy diff is -J(s_f - s_i)(sum of adj spins) - H(s_f - s_i),
        # using that s_f = -s_i
        ΔE = 2.0mc.spins[x, y] * (mc.J * adj_spin_sum + mc.H)
        # Probability of accepting spin flip (for ΔE ≤ 0 always accept)
        prob = exp(-ΔE / mc.T)

        if rand(ctx.rng) < prob
            mc.spins[x, y] *= -1
        end
    end
    return nothing
end

function Carlo.measure!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext)
    # Magnetization per lattice site
    mag = sum(mc.spins) / length(mc.spins)
    measure!(ctx, :Mag, mag)
    measure!(ctx, :AbsMag, abs(mag))
    measure!(ctx, :Mag2, mag^2)
    measure!(ctx, :Mag4, mag^4)

    Lx, Ly = size(mc.spins)
    energy = 0.0
    for x in 1:size(mc.spins, 1)
        for y in 1:size(mc.spins, 2)
            energy += -mc.J * mc.spins[x, y] *
                      (mc.spins[mod1(x+1, Lx), y] + mc.spins[x, mod1(y+1, Ly)])
            energy += -mc.H * mc.spins[x, y]
        end
    end
    energy /= length(mc.spins)
    measure!(ctx, :Energy, energy)
    measure!(ctx, :Energy2, energy^2)

    return nothing
end

function Carlo.register_evaluables(
    ::Type{HeisenHeatbathMC}, eval::AbstractEvaluator, params::AbstractDict
)
    T = params[:T]
    J = params[:J]
    N = params[:Lx] * params[:Ly]
    evaluate!(eval, :χ, (:AbsMag, :Mag2)) do mag, mag2
        return N * J/T * (mag2 - mag^2)
    end

    evaluate!(eval, :HeatCap, (:Energy2, :Energy)) do E2, E
        return N * (E2 - E^2) / T^2
    end

    evaluate!(eval, :BinderRatio, (:Mag2, :Mag4)) do mag2, mag4
        return 1 - mag4/(3mag2^2)
    end

    return nothing
end

function Carlo.write_checkpoint(mc::HeisenHeatbathMC, out::HDF5.Group)
    out["spins"] = mc.spins
    return nothing
end
function Carlo.read_checkpoint!(mc::HeisenHeatbathMC, in::HDF5.Group)
    mc.spins .= read(in, "spins")
    return nothing
end

end