
abstract type CN_Solver end
struct CN_CPU <: CN_Solver end
struct CN_GPU <: CN_Solver end
struct CN_Whole <: CN_Solver end
struct CN_Step <: CN_Solver end

# Tridiagonal elements
mutable struct TriDiag
    dl::Vector
    d::Vector
    du::Vector
end

mutable struct TriDiagGPU
    m::Int
    n::Int
    ldb::Int
    dl_gpu::ROCVector
    d_gpu::ROCVector
    du_gpu::ROCVector
    b_gpu::ROCVector
end

# CN Solver Objects
mutable struct CrankNicolson
    Nz::Int
    Nt::Int
    ν::Number
    σ::Number
    f::Function
    fu_old::Union{AbstractVector, Number}
    t::Number
    u::AbstractVector
    u_matrix::Union{AbstractVector, AbstractMatrix}
    A::AbstractMatrix
    B::Union{SparseMatrixCSC, Transpose{Float32, SparseMatrixCSC{Float32, Int64}}, Transpose{Float64, SparseMatrixCSC{Float64, Int64}}}
    ABC::Vector
end

# CN Parameters
mutable struct CN_Params{T}
    D::T
    Vp::T
    Vs::T
    Pp0::T
    Ps0::T
    Lp::T
    zp0::T
    Ls::T
    zs0::T
    boundary_conditions::Vector{String}

    function CN_Params{T}(sim_params::Sim_Params) where T
        @unpack c, τpump0, τlas0 = sim_params
        D = zero(T)
        Vp = -c
        Vs = -c
        Pp0 = 0.94 * (38e-3 / τpump0)
        Ps0 = 0.94 * (7.25e-4 / τlas0)
        Lp = 0.8493218 * (c * τpump0)
        zp0 = 128
        Ls = 0.8493218 * (c * τlas0)
        zs0 = 0.2
        boundary_conditions = ["dirichlet", "dirichlet"]
        new{T}(D, Vp, Vs, Pp0, Ps0, Lp, zp0, Ls, zs0, boundary_conditions)
    end
    
end

# CN Constructor function
function CN(sim_params::Sim_Params)
    @unpack Nz, Nt, width = sim_params
    type = typeof(width)
    nu = zero(type)
    sig = zero(type)
    f(u) = zero(type)
    fu_old = zeros(type, Nz)
    t = zero(type)
    u_matrix = zeros(type, Nz)
    A = zeros(type, Nz, 3)
    B = spdiagm(1 => zeros(type, Nz-1), 0 => zeros(type, Nz), -1 => zeros(type, Nz-1))
    ABC = zeros(type, Nz)
    CrankNicolson(Nz, Nt, nu, sig, f, fu_old, t, u_matrix, u_matrix, A, B, ABC)
end

# Space-time differentials that go in TriDiag matrix (i.e. Courant number-like)
function set_parameters(cn::CrankNicolson, D::Number, V::Number, Δt::Number, Δz::Number)
    type = typeof(V)

    cn.ν = V * Δt / (four(type) * Δz)
    cn.σ = D * Δt / (two(type) * Δz^2)
end

# Set initial vector to be solved
function set_uinit!(cn::CrankNicolson, u::Vector)
    cn.u .= u
end

# Solve Crank-Nicolson system over entire time axis on CPU or GPU
function solve(::CN_Whole, cn::CrankNicolson, tridiag::TriDiag, Δt)

    @unpack σ, f, Nt, fu_old, u, u_matrix, B = cn
    type = typeof(σ)
    for n in 1:Nt
        u_matrix[n, :] .= u
        fu = f(u)
        if n == 1
            fu_old = fu
        end
        u .= solve_banded(tridiag, cn, B * u .+ Δt .* (onep5(type) .* fu .- zerop5(type) .* fu_old))
        fu_old = fu
    end
    
end

function solve(::CN_Whole, cn::CrankNicolson, tridiag::TriDiagGPU, Δt)

    @unpack σ, f, Nt, fu_old, u, u_matrix, B = cn
    type = typeof(σ)
    for n in 1:Nt
        u_matrix[n, :] .= u
        fu = f(u)
        if n == 1
            fu_old = fu
        end
        u .= solve_banded(tridiag, B * u .+ Δt .* (onep5(type) .* fu .- zerop5(type) .* fu_old))
        fu_old = fu
    end
    
end

# Solve Crank-Nicolson system for one time step on CPU or GPU
function solve(::CN_Step, cn::CrankNicolson, tridiag::TriDiag, Δt)

    @unpack σ, f, t, fu_old, u, B = cn
    type = typeof(σ)

    fu = f(u)
    if t == 1
        fu_old .= fu
    end
    u .= solve_banded(tridiag, cn, B * u .+ Δt .* (onep5(type) .* fu .- zerop5(type) .* fu_old))
    fu_old .= fu


end

function solve(::CN_Step, cn::CrankNicolson, tridiag::TriDiagGPU, Δt)

    @unpack σ, f, t, fu_old, u, B = cn
    type = typeof(σ)

    fu = f(u)
    if t == 1
        fu_old = fu
    end
    u .= solve_banded(tridiag, B * u .+ Δt .* (onep5(type) .* fu .- zerop5(type) .* fu_old))
    fu_old = fu
end

function get_final_u(cn::CrankNicolson)
    return copy(cn.u_matrix[end, :])
end

function get_current_u(cn::CrankNicolson)
    return copy(cn.u)
end

# Fill elements of tridiagonal matrices A and B
function fillA_sp!(cn::CrankNicolson)
    @unpack σ, ν, A = cn
    # column-major ordering
    A[2:end, 1] .= -(σ - ν)  # subdiagonal
    A[:, 2] .= 1 + 2 * σ  # diagonal
    A[1:end-1, 3] .= -(σ + ν)  # superdiagonal
    A
end

function fillB_sp!(cn::CrankNicolson)
    @unpack σ, ν, Nz, B = cn
    type = typeof(σ)
    supdiag = (σ - ν) * ones(type, Nz-1)
    diag = (1 - 2*σ) * ones(type, Nz)
    subdiag = (σ + ν) * ones(type, Nz-1)
    B .= spdiagm(1 => supdiag, 0 => diag, -1 => subdiag)
    B
end

function setSparseArrays(cn::CrankNicolson)
    cn.A .= fillA_sp!(cn)
    cn.B .= fillB_sp!(cn)
end

function apply_boundary_conditions!(cn::CrankNicolson, boundary_conditions::AbstractVector)
    @unpack σ, ν, A, B = cn
    type = typeof(σ)
    for b in [1, 2]
        if boundary_conditions[b] == "dirichlet"
            # u(x,t) = 0
            
            if b == 1
                A[1,2] = one(type)
                A[2,3] = zero(type)
                B[1,1] = zero(type)
                B[2, 1] = zero(type)
            elseif b == 2
                A[end,2] = one(type)
                A[end-1,1] = zero(type)
                B[end,end] = zero(type)
                B[end-1,end] = zero(type)
            end

        elseif boundary_conditions[b] == "neumann"
            # u'(x,t) = 0
            if b == 1
                A[2,1] = -2*σ
                B[2,1] = 2*σ
            elseif b ==1 
                A[end-1, 3] = -2*σ
                B[end-1,end] = 2*σ
            end
        end
    end
end


# Tridiagonal solver on GPU (Float32 Only)
function sgtsv2!(tridiag::TriDiagGPU; pivoting::Bool=true)

    @unpack m, n, ldb, dl_gpu, d_gpu, du_gpu, b_gpu = tridiag
    (m ≤ 2) && throw(DimensionMismatch("The size of the linear system must be at least 3."))

    function bufferSize()
        out = Ref{Csize_t}(1)
        if pivoting
            AMDGPU.rocSPARSE.rocsparse_sgtsv_buffer_size(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, out)
        else
            AMDGPU.rocSPARSE.rocsparse_sgtsv_no_pivot_buffer_size(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, out)
        end
        return out[]
    end

    AMDGPU.rocSPARSE.with_workspace(bufferSize) do buffer
        if pivoting
            AMDGPU.rocSPARSE.rocsparse_sgtsv(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, buffer)
        else
            AMDGPU.rocSPARSE.rocsparse_sgtsv_no_pivot(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, buffer)
        end
    end

    B
end

# Tridiagonal solver on GPU (Float64 Only)
function dgtsv2!(tridiag::TriDiagGPU; pivoting::Bool=true)

    @unpack m, n, ldb, dl_gpu, d_gpu, du_gpu, b_gpu = tridiag
    (m ≤ 2) && throw(DimensionMismatch("The size of the linear system must be at least 3."))  

    function bufferSize()
        out = Ref{Csize_t}(1)
        if pivoting
            AMDGPU.rocSPARSE.rocsparse_dgtsv_buffer_size(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, out)
        else
            AMDGPU.rocSPARSE.rocsparse_dgtsv_no_pivot_buffer_size(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, out)
        end
        return out[]
    end

    AMDGPU.rocSPARSE.with_workspace(bufferSize) do buffer
        if pivoting
            AMDGPU.rocSPARSE.rocsparse_dgtsv(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, buffer)
        else
            AMDGPU.rocSPARSE.rocsparse_dgtsv_no_pivot(AMDGPU.rocSPARSE.handle(), m, n, dl_gpu, d_gpu, du_gpu, b_gpu, ldb, buffer)
        end
    end
    B
end


# Solve tridiagonal system of equations with CPU or GPU
function solve_banded(tridiag::TriDiagGPU, b::AbstractVector)
    @unpack b_gpu = tridiag
    type = typeof(b[1])
    copyto!(b_gpu, b)
    if type == Float32
        sgtsv2!(tridiag)
    else
        dgtsv2!(tridiag)
    end
    copyto!(b, b_gpu)

    return b
end

function solve_banded(tridiag::TriDiag, cn::CrankNicolson, b::AbstractVector)
    @unpack dl, d, du = tridiag
    dl .= cn.A[:, 1]
    d .= cn.A[:, 2]
    du .= cn.A[:, 3]
    LAPACK.gtsv!(dl[1:end-1], d, du[2:end], b)
    return b
end

# Propagate solution one step forward with CPU or GPU
function one_step!(cn::CrankNicolson, tridiag::TriDiag, t, Δt, u)
    cn.t = t
    solve(CN_Step(), cn, tridiag, Δt)
    u .= cn.u
    u
end

function one_step!(cn::CrankNicolson, tridiag::TriDiagGPU, t, Δt, u)
    cn.t = t
    solve(CN_Step(), cn, tridiag, Δt)
    u .= cn.u
    u
end

# Fixed inital source 
function gaussian(x, x0, w, A0)
    return A0 * exp.(-2 * (x .- x0).^2 / w^2)
end

# Time-dependent initial source
function gaussiant(t0, τ, A0)
    g(t) = A0 .* exp.(-2 .* ((t .- t0) ./ (0.8493218 .* τ)).^2)
    return g
end

# Function to artificially absorb pulse after crystal (i.e. ABC)
function absorber(z::Vector, width, strength; left=true, right=true)
    type = typeof(z[1])
    damping = ones(type, length(z))

    # Left-side absorbing boundary
    if left
        damping[1:width] .= exp.(strength * -(z[1:width] .- z[width]))
    end

    # Right-side absorbing boundary
    if right
        damping[end-width+1:end] .= exp.(strength * (z[end-width+1:end] .- z[end-width+1]))
    end

    return damping
end

# Create pump pulse CrankNicolson object 
function construct_pump(cn_params::CN_Params, sim_params::Sim_Params)
    @unpack D, Vp, zp0, Lp, Pp0, boundary_conditions = cn_params
    @unpack Nz, dt2, dz2, z2, τpump0, back = sim_params
    type = typeof(Pp0)

    # Main CN object including propagators
    pump = CN(sim_params)
    set_parameters(pump, D, Vp, dt2, dz2)
    setSparseArrays(pump)
    pump.B = transpose(pump.B)
    
    # Absorbing boundary (last relevant value is then at index "back")
    ab = absorber(z2, back, 1.0e3; left=false) # strength is carefully chosen to avoid divergence (adjust with caution)
    pump.ABC = ab

    # If you want whole pulse initialized at t=0
    #Pp = gaussian(z2, zp0, Lp, Pp0)

    # If you want time-dependent source
    Pp = zeros(type, Nz)
    gpt = gaussiant(3 * τpump0, τpump0, Pp0)
    set_uinit!(pump, Pp)

    apply_boundary_conditions!(pump, boundary_conditions)

    return pump, Pp, gpt
end

# Not really necessary because we use |A|^2 from GNLSE for seed power
function construct_seed(cn_params::CN_Params, sim_params::Sim_Params)
    @unpack D, Vs, zs0, Ls, Ps0, boundary_conditions = cn_params
    @unpack dt1, dz1, z1 = sim_params
    seed = CN(sim_params)
    set_parameters(seed, D, Vs, dt1, dz1)
    Ps = gaussian(z1, zs0, Ls, Ps0)
    set_uinit!(seed, Ps)
    apply_boundary_conditions!(seed, boundary_conditions)
    return seed, Ps
end

# This is to make sure everything is working okay (on CPU or GPU) before running DRE()
function time_evo(::CN_CPU, sim_params::Sim_Params)

    # Create CN solver objects
    @unpack Nz, Nt, dt2, dz2, front, back, c, τpump0, τlas0, z2, t2 = sim_params

    type = typeof(dt2)
    cn_param = CN_Params{type}(sim_params)
    @unpack D, Vp = cn_param

    @unpack Pp0, zp0, boundary_conditions = cn_param
    pump, Pp, gpt = construct_pump(cn_param, sim_params)
    tridiag = TriDiag(similar(Pp), similar(Pp), similar(Pp))

    pump2, Pp2, gpt2 = construct_pump(cn_param, sim_params)
    N2_1, gpt2 = getPrevPass(sim_params, "N2_1_RK4.csv", "Pow_out_1_RK4.csv")
    set_parameters(pump2, D, -Vp, dt2, dz2)
    setSparseArrays(pump2)
    pump2.B = transpose(pump2.B)
    pump2.ABC = reverse(pump.ABC)

    #println("Peak Pump Power : ", round(maximum(Pp), digits=2))
    l = 2^16
    nsteps = Int(2^14*781)
    
    for t = 0:Int(1e4*Nt-1)

        t_points1 = [t * dt2, (t - 1) * dt2, (t - 2) * dt2, (t - 3) * dt2]
        pump.u[11:14] .= Float32.(gpt.(t_points1))

        fp(u) = -pump.ABC .* abs.(u)
        pump.f = fp

        one_step!(pump, tridiag, t, dt2, Pp)

        if t*dt2 > 3e-9
            t_points2 = [(nsteps - (t - 3)) * dt2, (nsteps - (t - 2)) * dt2, (nsteps - (t - 1)) * dt2, (nsteps - t) * dt2]
            pump2.u[end-14:end-11] .= convert.(type, gpt2.(t_points2))

            fp2(u) = -pump2.ABC .* abs.(u)
            pump2.f = fp2
            one_step!(pump2, tridiag, t, dt2, Pp2)

        end
            
            
        if t % l == 0
            println(t, " steps taken")
            println(maximum(Pp))
            println(maximum(Pp2))
            p = plot(z2, Pp, lw=2, color="blue",)
            plot!(z2, Pp2, lw=2, color="navy",)
            plot!(z2, pump.ABC, lw=2, color="red")
            plot!(z2, pump2.ABC, lw=2, color="maroon")
            ylims!(-1.1*Pp0, 1.1*Pp0)
            #xlims!(0,0.001*Nz)
            vline!(dz2 .* [front, back], lw=3, color="black")
            vline!(dz2 .* [2660, 3071], lw=3, color="black")
            display(p)
        end
    end
end

function time_evo(::CN_GPU, sim_params::Sim_Params)

    # Create CN solver objects
    @unpack Nz, Nt, dt2, c, τpump0, τlas0, z2 = sim_params
    type = typeof(dt2)
    cn_param = CN_Params{type}(sim_params)
    @unpack Pp0 = cn_param
    pump, Pp = construct_pump(cn_param, sim_params)
    println("Peak Pump Power : ", round(maximum(Pp), digits=2))

    m = size(pump.A, 1)
    n = size(pump.A, 2)
    ldb = max(1, stride(pump.A, 2))
    dl_gpu, d_gpu, du_gpu = ROCArray.((pump.A[:, 1], pump.A[:, 2], pump.A[:, 3]))
    b_gpu = ROCArray{type}(undef, Nz)
    tridiag = TriDiagGPU{type}(m, n, ldb, dl_gpu, d_gpu, du_gpu, b_gpu)

    l = 200
    for t = 0:Nt-1
        one_step!(pump, tridiag, t, dt2, Pp)
        
        if t % l == 0
            println(t, " steps taken")
            #println("Available memory on GPU : ", AMDGPU.info()[1])
            #println("Used memory on GPU : ", AMDGPU.used())
            p = plot(z2, Pp, lw=2, color="blue", ylims=(0, 1.1*Pp0))
            #vline!(dz2 .* [front, back], lw=3, color="black")
            #plot!(seed.x_pts, soln2, lw=2, color="red")
            display(p)
        end
    end
end

println("CrankNicolson.jl compiled")

#using BenchmarkTools
#time_evo(CN_CPU(), sim_params)