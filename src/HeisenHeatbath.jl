module HeisenHeatbath

using Carlo
using HDF5
import Random, Random.AbstractRNG

export HeisenHeatbathMC

const Vector3D = NTuple{3, Float64}

# Note: Using temperature in units of energy (k_B = 1)
mutable struct HeisenHeatbathMC <: AbstractMC
    T::Float64 # Temperature
    J::Float64 # Interaction energy
    H::Float64 # External field
    spins::Matrix{Vector3D}
end

function HeisenHeatbathMC(params::AbstractDict)
    Lx, Ly = params[:Lx], params[:Ly]
    T = params[:T]
    J = params[:J]
    H = params[:H]
    return HeisenHeatbathMC(T, J, H, fill((0, 0, 0), Lx, Ly))
end

"""
    rand_vector([rng = default_rng()])

Generate a random 3D unit vector, uniformly distributed
"""
function rand_vector(rng::AbstractRNG = Random.default_rng())
    ϕ = 2π * rand(rng)
    θ = acos(2 * rand(rng) - 1)

    return (cos(ϕ)sin(θ), sin(ϕ)sin(θ), cos(θ))
end

function Carlo.init!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext, params::AbstractDict)
    if params[:rand_init]
        map!(_ -> rand_vector(ctx.rng), mc.spins, mc.spins)
    else
        fill!(mc.spins, (0, 0, 1))
    end
    return nothing
end

function Carlo.sweep!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext)
    Lx, Ly = size(mc.spins)
    for _ in 1:length(mc.spins)
        # Select site for spin change
        x = rand(ctx.rng, 1:Lx)
        y = rand(ctx.rng, 1:Ly)

        # Sum of nearest neighbors' spins
        adj_spin_sum = mc.spins[mod1(x-1, Lx), y] .+ mc.spins[x, mod1(y-1, Ly)] .+
                       mc.spins[mod1(x+1, Lx), y] .+ mc.spins[x, mod1(y+1, Ly)]
        H = adj_spin_sum ./ mc.T

        # Randomly generate new θ and ϕ according to Boltzmann distribution
        # (relative to adj_spin_sum)
        new_cosθ = log1p(rand(ctx.rng) * (exp(2H) - 1)) / H - 1
        new_ϕ = 2π * rand(ctx.rng)
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