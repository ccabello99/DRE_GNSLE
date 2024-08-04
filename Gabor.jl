
# Create object for performing Gabor transformations (i.e. STFT)
mutable struct Gabor{T}
    ng::Int64
    nt::Int64
    t_axis::Vector{T}
    Xgt_spec::Matrix{T}
    g::Vector{T}
    gt::Vector{T}
    ft::Vector{ComplexF64}
    tslide::Vector{T}
    window::Float64

    function Gabor{T}(sim_params::Sim_Params) where T
        @unpack t1, Nt, τ = sim_params
        ng = 200
        nt = Int(Nt / 2 + 1)
        t_axis = t1
        Xgt_spec = zeros(T, ng, Nt)
        g = zeros(T, Nt)
        gt = zeros(T, Nt)
        ft = zeros(ComplexF64, Nt)
        tslide = collect(range(t1[1], t1[end], ng))
        window = 0.1*τ
        new{T}(ng, nt, t_axis, Xgt_spec, g, gt, ft, tslide, window)
    end
end

# Transform over whole signal
function gabor_transform!(A::Vector, gabor::Gabor, fft_plan)
    @unpack ng = gabor
    for t in 1:ng
        gabor_transform!(A, gabor, fft_plan, t)
    end

end
    
# Transform at specific time
function gabor_transform!(A::Vector, gabor::Gabor, fft_plan, t::Int)

    @unpack Xgt_spec, t_axis, tslide, window, g, gt, ft, nt = gabor
    type = typeof(g[1])
    g .= exp.(-(t_axis .- tslide[t]).^2 / (window)^2)
    gt .= g .* real.(A)
    ft .= fftshift(fft_plan * gt)
    @inbounds Xgt_spec[t, :] .= convert.(type, abs.(ft)) / nt

end

# Transform over specific interval
function gabor_transform!(A::Vector, gabor::Gabor, fft_plan, int::Vector)

    for t in int[1]:int[end]
        gabor_transform!(A, gabor, fft_plan, t)
    end

end

println("Gabor.jl compiled")
