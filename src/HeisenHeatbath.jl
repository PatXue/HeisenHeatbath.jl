module HeisenHeatbath

using Carlo
using HDF5
using LinearAlgebra
import Random.default_rng, Random.AbstractRNG

export HeisenHeatbathMC

# Note: Using temperature in units of energy (k_B = 1)
struct HeisenHeatbathMC <: AbstractMC
    T::Float64 # Temperature
    J::Float64 # Interaction energy
    H::Float64 # External field
    spins::Array{Float64, 3}
end

function HeisenHeatbathMC(params::AbstractDict)
    Lx, Ly = params[:Lx], params[:Ly]
    T = params[:T]
    J = params[:J]
    H = params[:H]
    return HeisenHeatbathMC(T, J, H, fill(0, (Lx, Ly, 3)))
end

"""
    rand_vector(vec, [rng = default_rng()])

Fill the first 3 elements of vec with a 3D unit vector, generated uniformly
"""
function rand_vector!(vec, rng::AbstractRNG=default_rng())
    ϕ = 2π * rand(rng)
    θ = acos(2 * rand(rng) - 1)

    vec[1] = cos(ϕ)sin(θ)
    vec[2] = sin(ϕ)sin(θ)
    vec[3] = cos(θ)
    return nothing
end

function Carlo.init!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext, params::AbstractDict)
    vec::Vector{Float64} = [0, 0, 1]
    for slice in eachslice(mc.spins, dims=(1, 2))
        if params[:rand_init]
            rand_vector!(vec, ctx.rng)
        end
        slice .= vec
    end

    return nothing
end

function sum_adj(M, (x, y))
    Lx, Ly = size(M, 1), size(M, 2)
    return M[mod1(x-1, Lx), y, :] + M[x, mod1(y-1, Ly), :] +
           M[mod1(x+1, Lx), y, :] + M[x, mod1(y+1, Ly), :]
end

function Carlo.sweep!(mc::HeisenHeatbathMC, rng::AbstractRNG=default_rng())
    Lx, Ly = size(mc.spins, 1), size(mc.spins, 2)
    for _ in 1:length(mc.spins)
        # Select site for spin change
        x = rand(rng, 1:Lx)
        y = rand(rng, 1:Ly)

        # Sum of nearest neighbors' spins
        adj_spin_sum = sum_adj(mc.spins, (x, y))
        H = adj_spin_sum ./ mc.T
        unit_H = H ./ norm(H)
        H⊥ = nullspace(unit_H')

        # Randomly generate new θ and ϕ according to Boltzmann distribution
        # (relative to H)
        cosθ = log1p(rand(rng) * (exp(2norm(H)) - 1)) / norm(H) - 1
        ϕ = 2π * rand(rng)

        ϕ_comp = sqrt(1 - cosθ^2) * (cos(ϕ)H⊥[:, 1] + sin(ϕ)H⊥[:, 2])
        mc.spins[x, y, :] .= cosθ * unit_H + ϕ_comp
    end
    return nothing
end

function Carlo.sweep!(mc::HeisenHeatbathMC, ctx::Carlo.MCContext)
    Carlo.sweep!(mc, ctx.rng)
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