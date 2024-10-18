using Parameters, LaTeXStrings, DSP, CSV, DataFrames
using LinearAlgebra, FFTW, NumericalIntegration, Dierckx
using Plots, SparseArrays, AMDGPU

include("Sim_Param.jl")
include("Gabor.jl")
include("CrystalProperties.jl")
include("CrankNicolson.jl")
include("DRE.jl")


# This can be changed according to your machine
FFTW.set_num_threads(4)

# Nonlinear operator coefficients
struct NL_coeff
    n2::Function
    steep::ComplexF64
    γ::Vector
    NL::Vector{ComplexF64}
    function NL_coeff(sim_params, front, back)
        @unpack ω0, c, Aeff_s, λs, Nz = sim_params
        type = typeof(c)
        n2 = SaphKerrRefractiveIndex("n2_sapphire.csv")
        steep = 0 #1im / ω0
        γ = zeros(type, Nz)
        #γ[front:back] .= ω0 * n2(λs) / (c * Aeff_s) 
        γ[1:end] .= ω0 * n2(λs) / (c * Aeff_s) 
        NL = zeros(ComplexF64, Nz)
        NL .= 1im * γ
        new(n2, steep, γ, NL)
    end
end

# Unitary transformation into interaction picture
function IP_Transform!(ϕ::Vector, D̃::Vector, h)
    ϕ .= exp.(h .* D̃)
end

# Electric field temporal profile
function ComplexEnvelope(A0, t::Vector, t0::Number, z::Number, vg, ϕ, τ, GDD)
    C = sqrt(1 + (GDD / τ^2)^2)
    ϕ_σ = (1/2) * atan(GDD / τ^2)
    #T = similar(t)
    #T .= (t .- t0) .- z / vg
    A(T) = (A0 / sqrt(C)) .* exp.(1im .* (ϕ .+ ϕ_σ)) .* exp.(-(T.-t0).^2 / (2 .* (C .* τ)^2)) .* exp.(-1im .* (GDD / τ.^2) .* ((T.-t0).^2 ./ (2 .* (C .* τ)^2)))
    return A
end

# Take central-difference derivative of vector f with 2nd order boundary conditions
# too much error accumulation with self-steepening (switch to ForwardDiff.jl when possible TODO)
function df(f::Vector, dx)

    df = similar(f)

    df[1] = (-3*f[1] + 4*f[2] - f[3]) / (2*dx)
    df[2] = (f[3] - f[1]) / (2*dx)
    df[end-1] = (f[end] - f[end-2]) / (2*dx)
    df[end] = (3*f[end] - 4*f[end-1] + f[end-2]) / (2*dx)
    
    Threads.@threads for i in 3:length(f)-2
        @inbounds df[i] = (f[i+1] - f[i-1]) / (2*dx)
    end

    return df
    
end

# Take derivative of function f using ForwardDiff
function df(f::Function, x::Vector)
    return ForwardDiff.derivative.(f, x)
end

# Get pulse energy
function pulseEnergy(A::Vector, t::Vector)
    return NumericalIntegration.integrate(t, abs2.(A))
end 

# Get B-integral within crystal 
function calcBint(A2::Vector, z::Vector, n2::Number, λs::Number, front::Int, back::Int)
    Bint = (4π/λs) .* n2 .* NumericalIntegration.integrate(z[front:back], A2[front:back])
    return Bint
end

# Find index/indices for the maximum value of A
function findMaxIndex(A)
    return findfirst(x -> x == maximum(A), A)
end

# Get excited population from pumping stage with DRE
function getPumpedPop(N2csv::String, sim_params::Sim_Params)
    data = CSV.read(N2csv,DataFrame)
    N2_1 = data[!,1]
    z_ = collect(LinRange(-0.04, 0.04, 4096))
    spl1 = Spline1D(z_, N2_1, k=1)
    N2_(Z) = spl1(Z)

    #Shift to center of grid
    @unpack z1, dz1, front, back, width, Nz = sim_params
    type = typeof(z1[1])
    shift1 = Int(front+round(0.016 / dz1))
    shift2 = Int(Nz - (shift1 + width + 1))
    N2 = zeros(type, Nz)
    N2 = vcat(zeros(type, shift1), vcat(N2_.(z1)[front:back], zeros(type, shift2)))

    return N2
end

# Get excited population after a pass with GNLSE
function getPumpedPop(N2csv::String, z::Vector)
    data = CSV.read(N2csv,DataFrame)
    N2_1 = data[!,1]
    spl1 = Spline1D(z, N2_1, k=1)
    N2_(Z) = spl1(Z)

    return reverse(N2_.(z))
end


# Get field from last pass
function getField(At_csv::String, sim_params::Sim_Params)
    @unpack t1, Nz = sim_params
    data = CSV.read(At_csv,DataFrame)
    At_ = Vector{ComplexF64}(parse.(Complex{Float64}, data[!,1]))
    At_ ./= maximum(abs.(At_))

    #Shift to correct position
    At = vcat(At_[Int(Nz/2):Int(Nz)], At_[1:Int(Nz/2-1)])

    return At
end

# Get the instantaneous wavelength
function instWavelength(gabor::Gabor, A::Vector, t, f::Number, λ::Vector, sim_params::Sim_Params, fft_plan)

    @unpack freq, c, λs, front, width = sim_params

    # Windowed FFT at specific time t (can be over a specific interval)
    gabor_transform!(A, gabor, fft_plan, t)

    # Zero out half of freq. axis to avoid findMaxIndex() confusion
    gabor.Xgt_spec[:,Int(length(freq)/2):length(freq)] .= 0

    # Find current max wavelength(s) for region of interest
    diff = (c / λs) - (f/2π)
    wid = (width + 1) / 21
    p = 0
    for i in t[1]:t[end]
    index = findMaxIndex(gabor.Xgt_spec[i,:]')[2]
    @inbounds λ[Int(round(wid*p)+1):Int(round(wid*(p+1)))] .= c / abs(freq[index] .- diff)
    p += 1
    end
    

    return λ

end

# Update space-time gain function from excited population (only within crystal)
function updateGain!(A2::Vector, N2::Vector, σe::Vector, σa::Vector, front::Int, back::Int, t::Vector, 
    Esat::Vector, g::Vector, G::Vector, Nti::Number, λ, CC3::Number, Aeff_s::Number)

    # Space-time-dependent small signal gain coefficient for instantaneous wavelength
    g[front:back] .= (σe[front:back] .+ σa[front:back]) .* N2[front:back] .- σa[front:back] .* Nti

    # Wavelength-dependent saturation energy
    Esat .= CC3 ./ (λ .* (σe[front:back] .+ σa[front:back]))

    # Integral of squared norm of A w.r.t. time
    A_2 = NumericalIntegration.integrate(t[front:back], A2[front:back])

    # Gain function
    G[front:back] .= g[front:back] .* exp.((-Aeff_s ./ Esat) .* A_2)

end


# Linear operator in frequency domain
function LinOperator!(D̃::Vector, ω::Vector, β1, β2, β3, ω0)

    D̃ .= -1im .* (β1) .* (ω .- ω0) .+ 1im/2 .* β2 .* (ω .- ω0).^2 .- 1im/6 .* β3 .* (ω .- ω0).^3

end

# Nonlinear operator in temporal domain
function NLOperator(A::Vector, A2::Vector, G::Vector, h, dt, nl_coeff::NL_coeff)

    @unpack NL, steep = nl_coeff

    A2 .= abs2.(A)

    return h .* A .* (((G ./ 2) .+ 
                NL .* (A2 .+ (1im * steep) .* (df(A2, dt) .+ (df(A, dt) .* conj.(A))))))

end

# Create FFT plans which make future fft/ifft faster
function FFT_plans(u::Vector; inplace::Bool)
    if inplace
        fft_plan! = plan_fft!(u; flags=FFTW.MEASURE)
        ifft_plan! = plan_ifft!(u;flags=FFTW.MEASURE)
        return fft_plan!, ifft_plan!
    else
        fft_plan = plan_fft(u; flags=FFTW.MEASURE)
        ifft_plan = plan_ifft(u; flags=FFTW.MEASURE)
        return fft_plan, ifft_plan
    end
end


# Solve GNLSE for one space-time step using fourth-order Runge-Kutta method in the interaction picture
function RK4IP_step!(At::Vector, A2::Vector, A::Vector, ϕ::Vector, G::Vector, 
                        h::Number, dt::Number, nl_coeff::NL_coeff, fft_plan, ifft_plan)

    AI = similar(At)
    k1 = similar(At)
    k2 = similar(At)
    k3 = similar(At)
    k4 = similar(At)
    
    AI = ifft_plan * (ϕ .* fftshift((fft_plan * At)))
    k1 = ifft_plan * (ϕ .* fftshift((fft_plan * NLOperator(At, A2, G, h, dt, nl_coeff))))
    k2 = NLOperator(AI .+ k1 / 2, A2, G, h, dt, nl_coeff)
    k3 = NLOperator(AI .+ k2 / 2, A2, G, h, dt, nl_coeff)
    k4 = NLOperator(ifft_plan * (ϕ .* fftshift((fft_plan * (AI .+ k3)))), A2, G, h, dt, nl_coeff)

    A .= ifft_plan * (ϕ .* fftshift((fft_plan * (AI .+ (k1 .+ (2 .* (k2 .+ k3)) ./ 6))))) .+ k4 ./ 6

    
end

# Solve GNLSE and DRE for a seed pulse given pumped crystal population
function RK4IP(sim_params::Sim_Params, pass::Int, visualize::Bool)

    # Get simulation parameters
    @unpack (Nz, z1, dz1, Nt, t1, dt1, T1, t0, c, ϕ0, λs, c, Aeff_s,
            τlas0, τ, ϕ2, ω0, freq, front, back, width) = sim_params
    type = typeof(dt1)

    # Get constants
    phy_consts = PhyConsts{type}()
    @unpack Nti = phy_consts
    mult_consts = MultConsts{type}(phy_consts, sim_params)
    @unpack CC1, CC3 = mult_consts

    # Create objects for seed pulse
    cn_param = CN_Params{type}(sim_params)
    @unpack Ps0 = cn_param
    if pass == 1
        At = ComplexEnvelope(sqrt(Ps0), t1, t0, zero(type), c, ϕ0, τlas0, ϕ2)
    elseif pass == 2
        A_t = getField("At_1_RK4IP.csv", sim_params)
        data = CSV.read("Pow_out_1_RK4IP.csv",DataFrame)
        max_Pow = maximum(Vector(data[!,1]))
    elseif pass == 3
        A_t = getField("At_2_RK4IP.csv", sim_params)
        data = CSV.read("Pow_out_2_RK4IP.csv",DataFrame)
        max_Pow = maximum(Vector(data[!,1]))
    elseif pass == 4
        A_t = getField("At_3_RK4IP.csv", sim_params)
        data = CSV.read("Pow_out_3_RK4IP.csv",DataFrame)
        max_Pow = maximum(Vector(data[!,1]))
    elseif pass == 5
        A_t = getField("At_4_RK4IP.csv", sim_params)
        data = CSV.read("Pow_out_4_RK4IP.csv",DataFrame)
        max_Pow = maximum(Vector(data[!,1]))
    end

    # Shift initial pulse freq. to get unidirectional propagation
    f =(2π*1e14)
    if pass == 1
        A_t = At.(t1) .* exp.(1im .* f .* (t1.-t0))
    else
        A_t .*= sqrt(max_Pow)
    end
    println("FWHM duration of pulse : ", FWHM(t1, abs2.(A_t)))
    println("Pulse energy at time ", string(0), " ps : ", round(pulseEnergy(A_t, t1), digits=8))
    maxi = maximum(abs.(A_t))

    # For Gabor transform solver object
    gabor = Gabor{type}(sim_params)
    @unpack nt, ft = gabor

    # Plans to speed up (I)FFTs
    fft_plan, ifft_plan = FFT_plans(ComplexEnvelope(sqrt(Ps0), t1, t0, zero(type), c, ϕ0, τlas0, ϕ2).(t1); inplace = false);

    # For Gain updates
    G = zeros(type, Nz)
    g = zeros(type, Nz)
    Esat = zeros(type, Int(width+1))

    # For RK4IP steps
    A = similar(A_t)
    A2 = similar(G)
    A_half = similar(A_t)
    D̃ = similar(ft)
    ϕ = similar(ft)
    A2 .= abs2.(A_t)
    
    # Initialize crystal properties and populations
    σa, σe = getCrossSections("absorption_crosssection.csv", "emission_crosssection.csv")
    sapphire = SapphireSellmeier()
    n = SaphRefractiveIndex(sapphire)
    β0, β1, β2, β3 = SpectralPhases(n)
    vg = groupVel(β1)
    β_1 = ones(type, Nz) .* (1/c)
    β_2 = zeros(type, Nz)
    β_3 = zeros(type, Nz)

    atopop = AtoPop{type}(sim_params)
    if pumped
        if pass == 1
            N2_ = getPumpedPop("N2_1_RK4.csv", sim_params)    
        elseif pass == 2
            N2_ = getPumpedPop("N2_1_RK4IP.csv", z1)
        elseif pass == 3
            N2_ = getPumpedPop("N2_2_RK4IP.csv", z1)
        elseif pass == 4
            N2_ = getPumpedPop("N2_3_RK4IP.csv", z1)   
        elseif pass == 5
            N2_ = getPumpedPop("N2_4_RK4IP.csv", z1)
        end
    else
        N2_ = zeros(type, Nz)
    end
    
    atopop.N2 .= N2_
    atopop.N1 .= Nti .- atopop.N2
    
    # Get initial instantaneous wavelength
    λ = zeros(type, Int(width+1))
    λ .= instWavelength(gabor, A_t, collect(90:110), f, λ, sim_params, fft_plan)

    # Create seed power arrays for population update
    Ps = similar(G)
    Ps .= abs2.(A_t)
    maxi2 = maximum(Ps)

    # Initialize arrays of cross-sections
    
    σ_a = zeros(type, Nz)
    σ_e = zeros(type, Nz)
    front = Int(front+round(0.016 / dz1) + 1) # This can change depending on how you set it up
    back = Int(front + width)
    for i in front:back
        @inbounds σ_a[i] = σa(λ[Int(i%front + 1)])
        @inbounds σ_e[i] = σe(λ[Int(i%front + 1)])
    end

    # Instantaneous spectral phase within crystal
    β_1[front:back] .= β1(λs)
    β_2[front:back] .= β2(λs)
    β_3[front:back] .= β3(λs)
    
    # Set initial gain function
    nl_coeff = NL_coeff(sim_params, front, back)
    updateGain!(Ps, atopop.N2, σ_e, σ_a, front, back, t1, Esat, g, G, Nti, λ, CC3, Aeff_s)
    

    # Create RK4IP solver objects
    nl_coeff = NL_coeff(sim_params, front, back)
    ω = BLAS.scal(2π, freq)
    LinOperator!(D̃, ω, β_1, β_2, β_3, f)
    IP_Transform!(ϕ, D̃, dz1/2)

    # Initial B-integral equal to zero
    Bint = 0
    
    # Some arrays/values relevant for plotting/later processing
    k = 100 # Change this as wanted
    diff = (c / λs) - (f/2π)
    max_N2 = zeros(type, Int(round(Nt/2 / k)))
    Pow_in = zeros(type, Int(round(Nt/2 / k)))
    Pow_out = zeros(type, Int(round(Nt/2 / k)))
    Energy_t = zeros(type, Int(round(Nt/2 / k)))
    Bint_t = zeros(type, Int(round(Nt/2 / k)))
    SpectPow_t = zeros(type, Int(round(Nt/2 / k)), Int((5*sim_params.Nt/8) - (Nt/2)))
    SpectPhase_t = zeros(type, Int(round(Nt/2 / k)), Int((5*sim_params.Nt/8) - (Nt/2)))
    
    fft_A = fftshift((fft_plan * A_t))
    maxif = maximum(abs.(fft_A))

   
    # Main evolution loop
    for t in 0:Int(Nt/8)

        # Update instantaneous wavelength and cross-section arrays
        
        if t % 1317 == 0
            λ .= instWavelength(gabor, A_t, collect(90:110), f, λ, sim_params, fft_plan)
            for i in front:back
                @inbounds σ_a[i] = σa(λ[Int(i%front + 1)])
                @inbounds σ_e[i] = σe(λ[Int(i%front + 1)])
            end
        end

        # Update complex envelope
        RK4IP_step!(A_t, A2, A, ϕ, G,  dz1, dt1, nl_coeff, fft_plan, ifft_plan,)
        A_t .= A
        β_1[front:back] .= β1.(λ)
        β_2[front:back] .= β2.(λ)
        β_3[front:back] .= β3.(λ)
        LinOperator!(D̃, ω, β_1, β_2, β_3, f)
        IP_Transform!(ϕ, D̃, dz1/2)

        # Update crystal populations
        Ps .= abs2.(A_t)
        AtoPopUpdate!(Seed(), atopop, sim_params, mult_consts, phy_consts, σ_a, σ_e, Ps, front, back, λ[1])

        # Update space-time gain function
        updateGain!(Ps, atopop.N2, σ_e, σ_a, front, back, t1, Esat, g, G, Nti, λ, CC3, Aeff_s)

        # Update B-integral
        Bint += calcBint(Ps, z1, nl_coeff.n2(λs), λs, front, back)

        # Save data for later processing and/or plot stuff
        
        if t % k == 0
            println("Pulse energy at time ", round(t*dt1*1e12, digits=3), " ps : ", round(pulseEnergy(A_t, t1), digits=8))

            # Data to save
            SpectPow_t[Int(t / k), :] .= abs.(fft_a[Int(Nt/2):Int(5*sim_params.Nt/8 - 1)])
            SpectPhase_t[Int(t / k), :] .= fft_phase[Int(Nt/2):Int(5*sim_params.Nt/8 - 1)]
            max_N2[Int(t / k)] = maximum(atopop.N2)
            Pow_in[Int(t / k)] = Ps[front]
            Pow_out[Int(t / k)] = Ps[back]
            Energy_t[Int(t / k)] = round(pulseEnergy(A_t, t1), digits=12)
            Bint_t[Int(t / k)] = Bint

            if visualize
                #p = plot(z1, real.(A_t) ./ maxi, label=L"Re[A(t)]", lw=2, color="blue", ylims = (-1.25, 1.25))
                #plot!(z1, imag.(A_t) ./ maxi, label=L"Im[A(t)]", lw=2, color="green")
                fft_a = fftshift(fft_plan * A_t)
                fft_phase = DSP.Unwrap.unwrap(angle.(fft_a)) .- ((freq).*-4.1887249e-10) .+ 103100 .+ 5.515725806256307*t .+ (freq .* 22.56107406e-15 * t)
                #println(fft_phase[1])
                #println(fft_phase[99717])
                extra = fft_phase[92221] # This can change
                # Figures to save
                
                #p = plot(z1.*1e3, abs2.(A_t) ./ maxi2, label=L"|A(t)|^2", lw=2, color="red", xlabel="z (mm)", ylabel="Normalized Peak Power (arb. u)")
                #yaxis2 = twinx()
                #plot!(yaxis2, z1.*1e3, atopop.N2 ./ Nti, color="black", lw=2, label="Excited Ti^3+ Ions", legend=:right, ylims=(0,1), ylabel="Fraction of Excited Ions (arb. u)")
                #display(p)
                #savefig("RK4IP_GDD+TOD+DRE_"*string(pass)*"_t"*string(t)*".png")
                p = plot((freq .+ diff) .* 1e-12, abs.(fft_a) ./ maxif, label="Spectral Power", lw=2, color="red", xlabel="Frequency (THz)", ylabel="Normalized Spectral Power (arb. u)", legend=:topleft, xlims=(360, 400), ylims=(0,1.1))
                yaxis2 = twinx()
                plot!(yaxis2, (freq .+ diff) .* 1e-12, (fft_phase) .- extra,  label = "Spectral Phase", ylabel="Normalized Spectral Phase (rad)", lw=2, color="blue", legend=:topright, xlims=(360, 400), ylims=(-20, 20)) #
                display(p)
                #savefig("RK4IP_GDD+TOD+DRE_"*string(pass)*"_spect_t"*string(t)*".png")
                savefig("RK4IP_GDD+TOD_"*string(pass)*"_spect_t"*string(t)*".png")
            end        
        end

    end
    
    # Write saved data points to CSV files
    
    if pass == 1
        CSV.write("N2_1_RK4IP.csv", Tables.table(atopop.N2), writeheader=true)
        CSV.write("Ps_1_RK4IP.csv", Tables.table(Ps), writeheader=true)
        CSV.write("At_1_RK4IP.csv", Tables.table(A_t), writeheader=true)
        CSV.write("fftA_1_RK4IP.csv", Tables.table(SpectPow_t), writeheader=true)
        CSV.write("fftPhase_1_RK4IP.csv", Tables.table(SpectPhase_t), writeheader=true)
        CSV.write("max_N2_1_RK4IP.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_1_RK4IP.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_1_RK4IP.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("En_t_1_RK4IP.csv", Tables.table(Energy_t), writeheader=true)
        CSV.write("Bint_t_1_RK4IP.csv", Tables.table(Bint_t), writeheader=true)
    elseif pass == 2
        CSV.write("N2_2_RK4IP.csv", Tables.table(atopop.N2), writeheader=true)
        CSV.write("Ps_2_RK4IP.csv", Tables.table(Ps), writeheader=true)
        CSV.write("At_2_RK4IP.csv", Tables.table(A_t), writeheader=true)
        CSV.write("fftA_2_RK4IP.csv", Tables.table(SpectPow_t), writeheader=true)
        CSV.write("fftPhase_2_RK4IP.csv", Tables.table(SpectPhase_t), writeheader=true)
        CSV.write("max_N2_2_RK4IP.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_2_RK4IP.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_2_RK4IP.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("En_t_2_RK4IP.csv", Tables.table(Energy_t), writeheader=true)
        CSV.write("Bint_t_2_RK4IP.csv", Tables.table(Bint_t), writeheader=true)
    elseif pass == 3
        CSV.write("N2_3_RK4IP.csv", Tables.table(atopop.N2), writeheader=true)
        CSV.write("Ps_3_RK4IP.csv", Tables.table(Ps), writeheader=true)
        CSV.write("At_3_RK4IP.csv", Tables.table(A_t), writeheader=true)
        CSV.write("fftA_3_RK4IP.csv", Tables.table(SpectPow_t), writeheader=true)
        CSV.write("fftPhase_3_RK4IP.csv", Tables.table(SpectPhase_t), writeheader=true)
        CSV.write("max_N2_3_RK4IP.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_3_RK4IP.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_3_RK4IP.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("En_t_3_RK4IP.csv", Tables.table(Energy_t), writeheader=true)
        CSV.write("Bint_t_3_RK4IP.csv", Tables.table(Bint_t), writeheader=true)
    elseif pass == 4
        CSV.write("N2_4_RK4IP.csv", Tables.table(atopop.N2), writeheader=true)
        CSV.write("Ps_4_RK4IP.csv", Tables.table(Ps), writeheader=true)
        CSV.write("At_4_RK4IP.csv", Tables.table(A_t), writeheader=true)
        CSV.write("fftA_4_RK4IP.csv", Tables.table(SpectPow_t), writeheader=true)
        CSV.write("fftPhase_4_RK4IP.csv", Tables.table(SpectPhase_t), writeheader=true)
        CSV.write("max_N2_4_RK4IP.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_4_RK4IP.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_4_RK4IP.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("En_t_4_RK4IP.csv", Tables.table(Energy_t), writeheader=true)
        CSV.write("Bint_t_4_RK4IP.csv", Tables.table(Bint_t), writeheader=true)
    elseif pass == 5
        CSV.write("N2_5_RK4IP.csv", Tables.table(atopop.N2), writeheader=true)
        CSV.write("Ps_5_RK4IP.csv", Tables.table(Ps), writeheader=true)
        CSV.write("At_5_RK4IP.csv", Tables.table(A_t), writeheader=true)
        CSV.write("fftA_5_RK4IP.csv", Tables.table(SpectPow_t), writeheader=true)
        CSV.write("fftPhase_5_RK4IP.csv", Tables.table(SpectPhase_t), writeheader=true)
        CSV.write("max_N2_5_RK4IP.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_5_RK4IP.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_5_RK4IP.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("En_t_5_RK4IP.csv", Tables.table(Energy_t), writeheader=true)
        CSV.write("Bint_t_5_RK4IP.csv", Tables.table(Bint_t), writeheader=true)
    end
    
    
end


#=
Different visualization scripts 


@unpack Nz, z1, dz1, Nt, t1, dt1, T1, t0, λs, c, ϕ0, τlas0, τ, ϕ2, ω0, freq, front, back = sim_params
type = typeof(dt1)

# Get constants
phy_consts = PhyConsts{type}()
mult_consts = MultConsts{type}(phy_consts, sim_params)
@unpack CC1, CC3 = mult_consts


# Create objects for seed pulse
cn_param = CN_Params{type}(sim_params)
@unpack Ps0 = cn_param
At = ComplexEnvelope(sqrt(Ps0), t1, t0, zero(type), c, ϕ0, (τlas0), ϕ2);


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (ps)",
             ylabel="Normalized Field Amplitude", title="30 fs (FWHM) pulse with 208000 fs^2 GDD")
lines!(ax, t1.*1e12, real.(A), color="blue", label="Carrier Wave")
lines!(ax, t1.*1e12, abs.(A), color="red", label="Envelope")
lines!(ax, t1.*1e12, -abs.(A), color="red")
xlims!(10, 50)
axislegend(ax, position = :rt)
fig

#Makie.save("highly_chirped_pulse.png", fig)


using CairoMakie

# For Gabor transform solver object
#sim_params.t1 = collect(range(-0.5e-12, 0.5e-12, step=0.1e-15))
gabor = Gabor{type}(sim_params)
@unpack nt, tslide, t_axis = gabor

# Plans to speed up (I)FFTs
fft_plan, ifft_plan = FFT_plans(At.(t_axis); inplace = false);

# Transform
#f = (2π*1e14)
f = ω0
A_t = (At.(t1) .* exp.(1im .* -f .* t1))
maxi = maximum(abs.(At.(t1)))
gabor_transform!(A_t, gabor, fft_plan)
#Xgt = gabor.Xgt_spec[:, 41500:51000]
#gabor.Xgt_spec[:, 4000:5000] .= 0

@unpack freq = sim_params
#freqq = freq[41500:51000]

#using Plots

#display(heatmap(tslide, freq, gabor.Xgt_spec'))


fig = Figure(fontsize = 48, size=(1920,1080))
ax = Axis(fig[1, 1], 
xlabel = L"\textbf{Time (fs)}", ylabelpadding = 40, 
ylabel = L"\textbf{Frequency (THz)}", limits=(nothing, nothing, 350, 410),
title = L"\textbf{Gabor Transform of Seed Pulse }(\phi_{2} = 1500 \text{ fs}^{2})")

#hm = CairoMakie.heatmap!(fig[1, 1], tslide.*1e12, freq.*1e-12, Xgt_spec;
#        colormap = :thermal)
#diff = (c / λs) - (2e12)
diff = 0
hm = CairoMakie.heatmap!(ax, tslide.*1e15, (freq .+diff).*1e-12, gabor.Xgt_spec ./ (maximum(1.39414871215918 .*gabor.Xgt_spec)), colormap = :vikO100, interpolate=true)
#lines!(ax, [tslide[75], tslide[75]].*1e12, [350, 420], color=:white, linewidth=4, linestyle=:dash)
#lines!(ax, [tslide[125], tslide[125]].*1e12, [350, 420], color=:white, linewidth=4, linestyle=:dash)

cb = CairoMakie.Colorbar(fig[1, 2], hm, size=30, ticks=[0, 0.2, 0.4, 0.6];
        label = L"\textbf{Normalized Spectral Power (arb. u.)}")

fig

=#