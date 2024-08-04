using Parameters

# Helper functions
onep5(::Type{T}) where {T<:Number} = convert(T,1.5)
zerop5(::Type{T}) where {T<:Number} = convert(T,0.5)
four(::Type{T}) where {T<:Number} = convert(T,4.0)
two(::Type{T}) where {T<:Number} = convert(T,2.0)

@with_kw struct Sim_Params{T}
    # Size of the grid
    Nz::Int64
    Lz1::T = 0.08
    Lz2::T = 0.08
    dz1::T = Lz1 / Nz
    dz2::T = Lz2 / Nz
    z1::Vector{T} = collect(LinRange(-Nz/2, Nz/2, Nz) .* dz1)
    z2::Vector{T} = collect(LinRange(1, Nz, Nz) .* dz2)

    # Crystal grid region
    width::T = round(8e-3 / dz2)
    front::Int64 = Int64(round((0.25*Nz)))
    back::Int64 = Int64((front+width))

    # Pulse durations (FWHM)
    τlas0::T = 30e-15
    τpump0::T = 150e-9

    # Simulation times
    c::T = 2.998e8
    Nt::Int64 = Nz
    dt1::T = dz1 / (1 * c)
    T1::T = 8e-3 / c
    dt2::T = dz2 / (1 * c)
    T2::T = 197 / c
    t1::Vector{T} = collect(LinRange(-Nt/2, Nt/2, Nt) .* dt1)
    t2::Vector{T} = collect(LinRange(1, Nt, Nt) .* dt2)

    # Frequency domain
    fs::T = 1 / dt1
    freq::Vector{T} = (fs / Nt) .* range(-Nt/2, Nt/2, Nt)

    # Pulse properties
    λs::T = 785e-9
    λk::T = 1000e-9
    λp::T = 527e-9
    ω0::T = 2π * (c / λs)
    ϕ0::T = zero(T)
    #ϕ2::T = zero(T)
    #ϕ2::T = 1000e-30
    ϕ2::T = 208000e-30
    Aeff_p::T = π * (2000e-6)^2 / 2
    Aeff_s::T = π * (1200e-6)^2 / 2
    #Aeff_s::T = π * (120000e-6)^2 / 2
    τ::T = τlas0 * √(1 + (4 * log(2) * ϕ2 / (τlas0^2))^2)
    t0::T = -6.67e-11
    #t0::T = -6.005e-11 #zero(T)
end

# Compute FWHM of pulse (Y) given its x-axis (X)
function FWHM(X, Y)
    half_max = maximum(Y) / 2
    d = sign.(half_max .- Y[1:end-1]) .- sign.(half_max .- Y[2:end])
    left_idx = findfirst(d .> 0)
    right_idx = findlast(d .< 0)
    return X[right_idx] - X[left_idx]
end

sim_params = Sim_Params{Float64}(Nz=2^18)

println("Nz : ", sim_params.Nz)
println("Nt : ", sim_params.Nt)