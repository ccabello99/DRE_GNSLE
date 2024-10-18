
struct Pump end
struct Seed end

# Physical Constants (SI units)
struct PhyConsts{T}
    h::T
    Nti::T
    τ2::T
    Tc::T
    αg::T
    Δλ::T
    function PhyConsts{T}() where T
        h = 1.055e-34 # Planck's constant
        Nti = 3.73e25 # Ti concentration 
        τ2 = 3.2e-6 # upper level population lifetime
        Tc = 200 # Crystal temp 
        αg = 0.00 # Generic losses
        Δλ = 1e-9 # Spontaneous emission wavelength resolutions
        new{T}(h, Nti, τ2, Tc, αg, Δλ)
    end
end

# Atomic Populations
mutable struct AtoPop{T}
    N1::Vector{T}
    N2::Vector{T}
    function AtoPop{T}(sim_params::Sim_Params) where T 
        @unpack Nz = sim_params
        N1 = zeros(T, Nz)
        N2 = zeros(T, Nz)
        new{T}(N1, N2)
    end
end

# Use excited population and pump pulse from previous pass saved data
function getPrevPass(sim_params::Sim_Params, N2csv::String, Ppcsv::String)
    @unpack Nz, front, back, dt2 = sim_params

    data = CSV.read(N2csv,DataFrame)
    prev_pass = reverse(data[!,1][front:back])
    type = typeof(prev_pass[1])
    N2 = vcat(zeros(type, front-1), vcat(prev_pass, zeros(type, (Nz - back))))


    data = CSV.read(Ppcsv,DataFrame)
    Pp1p = reverse(data[!, 1])
    times = range(0, length(Pp1p), length(Pp1p)) .* 2^14 * dt2
    spl1 = Spline1D(times, Pp1p, k=3)
    Pp_1p(t) = spl1(t)

    return N2, Pp_1p
end


# Multiplicative constants
struct MultConsts{T}
    CC1::T 
    CC2::T 
    CC3::T
    
    function MultConsts{T}(pconst::PhyConsts, sim_params::Sim_Params) where T
        @unpack h, Δλ = pconst
        @unpack c, Aeff_s = sim_params
        CC1 = 1 / (h * c)
        CC2 = (2*h*c^2 * Δλ)
        CC3 = (Aeff_s * h * c)
        new{T}(CC1, CC2, CC3)
    end
end


# Update atomic populations in crystal only for pump or seed pulse
function AtoPopUpdate!(::Pump, atopop::AtoPop, sim_params::Sim_Params, multconst::MultConsts, phyconst::PhyConsts, σa::Function, σe::Function, P::Vector, front::Int, back::Int)
    @unpack N1, N2 = atopop
    @unpack λp, dt2, Aeff_p = sim_params
    @unpack CC1 = multconst
    @unpack Nti, τ2 = phyconst 

    # Time derivative of excited population
    dN2_dt(N2_c, N1_c, P_c) = (CC1 / Aeff_p .* (λp .* (σa(λp) .* N1_c .- σe(λp) .* N2_c) .* abs.(P_c))) .- N2_c ./ τ2

    # Only update within crystal
    N1_crys = atopop.N1[front:back]
    N2_crys = atopop.N2[front:back]
    P_crys = P[front:back]

    # Current ground state population
    N1_crys .= Nti .- N2_crys

    # Runge-Kutta 4 step
    k1 = dN2_dt(N2_crys, N1_crys, P_crys)
    k2 = dN2_dt(N2_crys .+ 0.5 .* dt2 .* k1, N1_crys, P_crys)
    k3 = dN2_dt(N2_crys .+ 0.5 .* dt2 .* k2, N1_crys, P_crys)
    k4 = dN2_dt(N2_crys .+ dt2 .* k3, N1_crys, P_crys)  
    N2_crys .= N2_crys .+ (dt2 / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)

    # Assign current populations
    N1_crys .= Nti .- N2_crys

    atopop.N1[front:back] .= N1_crys
    atopop.N2[front:back] .= N2_crys

end

function AtoPopUpdate!(::Seed, atopop::AtoPop, sim_params::Sim_Params, multconst::MultConsts, phyconst::PhyConsts, σa::Vector, σe::Vector, P::Vector, front::Int, back::Int, λ)
    @unpack N1, N2 = atopop
    @unpack dt1, Aeff_s = sim_params
    @unpack CC1 = multconst
    @unpack Nti, τ2 = phyconst 

    # Time derivative of excited population
    dN2_dt(N2_c, N1_c, P_c) = (CC1 / Aeff_s .* (λ .* (σa[front:back] .* N1_c .- σe[front:back] .* N2_c) .* abs.(P_c))) .- N2_c ./ τ2

    # Only update within crystal
    N1_crys = atopop.N1[front:back]
    N2_crys = atopop.N2[front:back]
    P_crys = P[front:back]

    # Current ground state population
    N1_crys .= Nti .- N2_crys

    # Runge-Kutta 4 step
    k1 = dN2_dt(N2_crys, N1_crys, P_crys)
    k2 = dN2_dt(N2_crys .+ 0.5 .* dt1 .* k1, N1_crys, P_crys)
    k3 = dN2_dt(N2_crys .+ 0.5 .* dt1 .* k2, N1_crys, P_crys)
    k4 = dN2_dt(N2_crys .+ dt1 .* k3, N1_crys, P_crys)  
    N2_crys .= N2_crys .+ (dt1 / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)

    # Assign current populations
    N1_crys .= Nti .- N2_crys

    atopop.N1[front:back] .= N1_crys
    atopop.N2[front:back] .= N2_crys

end


# Solve the Dynamical Rate Equations for pump or seed
function DRE(::Pump, sim_params::Sim_Params, pass::Int)

    # Get constants and atomic populations
    @unpack Nt, Nz, dt2, dz2, λp, z2, front, back, c = sim_params
    type = typeof(dt2)
    atopop1 = AtoPop{type}(sim_params)
    atopop2 = AtoPop{type}(sim_params)
    pconst = PhyConsts{type}()
    multconst = MultConsts{type}(pconst, sim_params)

    # Construct CrankNicolson solver object
    cn_param = CN_Params{type}(sim_params)
    @unpack Pp0, D, Vp = cn_param
    @unpack Pp0, zp0, boundary_conditions = cn_param
    pump1, Pp1, gpt = construct_pump(cn_param, sim_params)
    tridiag = TriDiag(similar(Pp1), similar(Pp1), similar(Pp1))

    # Create arrays for later processing
    k = 2^14
    nsteps = Int(2^14*780)
    max_N2 = zeros(type, Int(round(nsteps / k))+1)
    Pow_in = zeros(type, Int(round(nsteps / k))+1)
    Pow_out = zeros(type, Int(round(nsteps / k))+1)

    # Get crystal properties
    σa, σe = getCrossSections("absorption_crosssection.csv", "emission_crosssection.csv")
    sapphire = SapphireSellmeier()
    n = SaphRefractiveIndex(sapphire)
    β0, β1, β2, β3 = SpectralPhases(n)
    vg = groupVel(β1)

    # Create second pump pulse for reflection
    if pass == 2
        pump2, Pp2, gpt2 = construct_pump(cn_param, sim_params)
        N2_1, gpt2 = getPrevPass(sim_params, "N2_1_RK4.csv", "Pow_out_RK4.csv")
        set_parameters(pump2, D, -Vp, dt2, dz2)
        setSparseArrays(pump2)
        pump2.B = transpose(pump2.B)
        pump2.ABC = reverse(pump1.ABC)
        Pow_in_ref = zeros(type, Int(round(nsteps / k))+1)
        Pow_out_ref = zeros(type, Int(round(nsteps / k))+1)
    end

    for t in 0:nsteps

        # Source injection
        t_points = [t * dt2, (t - 1) * dt2, (t - 2) * dt2, (t - 3) * dt2, (t - 4) * dt2, (t - 5) * dt2, (t - 6) * dt2, (t - 7) * dt2, (t - 8) * dt2, (t - 9) * dt2]
        pump1.u[11:20] .= convert.(type, gpt.(t_points))

        # Update populations
        AtoPopUpdate!(Pump(), atopop1, sim_params, multconst, pconst, σa, σe, Pp1, front, back)
        
        # Pump terms
        fp(u) = vg(λp) .* ((σe(λp) .* atopop1.N2 .- σa(λp) .* atopop1.N1) .* abs.(u)) .- (pump1.ABC .* abs.(u))
        pump1.f = fp
        one_step!(pump1, tridiag, t, dt2, Pp1)

        # Launch reflected pump pulse after 3 ns (~1m propagation)
        if t*dt2 > 3e-9
            t_points2 = [(nsteps - (t - 9)) * dt2, (nsteps - (t - 8)) * dt2, (nsteps - (t - 7)) * dt2, (nsteps - (t - 6)) * dt2, (nsteps - (t - 5)) * dt2, (nsteps - (t - 4)) * dt2, (nsteps - (t - 3)) * dt2, (nsteps - (t - 2)) * dt2, (nsteps - (t - 1)) * dt2, (nsteps - t) * dt2]
            pump2.u[end-20:end-11] .= convert.(type, gpt2.(t_points2))

            atopop2.N2 = vcat(zeros(type, 2660), vcat(atopop1.N2[front:back], zeros(type, (Nz-3071))))
            AtoPopUpdate!(Pump(), atopop2, sim_params, multconst, pconst, σa, σe, Pp2, 2661, 3071)

            fp2(u) = vg(λp) .* ((σe(λp) .* atopop2.N2 .- σa(λp) .* atopop2.N1) .* abs.(u)) .- (pump2.ABC .* abs.(u))
            pump2.f = fp2
            one_step!(pump2, tridiag, t, dt2, Pp2)

            atopop1.N1[front:back] .= atopop2.N1[2661:3071]
            atopop1.N2[front:back] .= atopop2.N2[2661:3071]
        end


        # Save data for later processing and/or plot
        if t % k == 0
            #p = plot(z2, Pp1, lw=2, color="green", label="Initial Pump", ylabel="Peak Power (W)", xlabel="Position (m)", legend= :topleft, ylims=(0, 1.5*Pp0))
            #plot!(p, z2, Pp2, lw=2, color="olive", label="Reflected Pump")
            #yaxis2 = twinx()
            #plot!(yaxis2, z2, atopop1.N2 .+ 1e-12, lw=2, color="red", label="Excited Ions", ylabel="Population (1/m³)", ylims=(1,1e30), legend= :topright, yscale=:log10)
            #plot!(yaxis2, z2, atopop2.N2 .+ 1e-12, lw=2, color="maroon", label="Excited Ions for Reflection", yscale=:log10)
            #savefig("pumping_stage"*string(t))
            println(t, " steps taken")
            max_N2[Int(t / k) + 1] = maximum(atopop1.N2)
            Pow_in[Int(t / k) + 1] = Pp1[front]
            Pow_out[Int(t / k) + 1] = Pp1[back]
            Pow_in_ref[Int(t / k) + 1] = Pp2[3071]
            Pow_out_ref[Int(t / k) + 1] = Pp2[2661]
            #ylims!(0, 1)
            #display(p)
        end

    end

    # Write saved data points to CSV files
    if pass == 1
        CSV.write("N2_1_RK4.csv", Tables.table(N2), writeheader=true)
        CSV.write("max_N2_1_RK4.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_1_RK4.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_1_RK4.csv", Tables.table(Pow_out), writeheader=true)
    elseif pass == 2
        CSV.write("N2_RK4.csv", Tables.table(atopop1.N2), writeheader=true)
        CSV.write("max_N2_RK4.csv", Tables.table(max_N2), writeheader=true)
        CSV.write("Pow_in_RK4.csv", Tables.table(Pow_in), writeheader=true)
        CSV.write("Pow_out_RK4.csv", Tables.table(Pow_out), writeheader=true)
        CSV.write("Pow_in_ref_RK4.csv", Tables.table(Pow_in_ref), writeheader=true)
        CSV.write("Pow_out_ref_RK4.csv", Tables.table(Pow_out_ref), writeheader=true)
    end
end

# Not really necessary in current framework (i.e. we only need to use AtoPopUpdate(Seed))
function DRE(::Seed, sim_params::Sim_Params)

    @unpack Nt, λs, dt1 = sim_params
    type = typeof(dt1)
    atopop = AtoPop{type}(sim_params)
    pconst = PhyConsts{type}()
    multconst = MultConsts{type}(pconst, sim_params)

    @unpack N1, N2 = atopop
    @unpack Nt, τ2 = pconst
    @unpack CC1 = multconst
    
    cn_param = CN_params{type}(sim_params)
    seed, Ps = construct_seed(cn_param, sim_params)
    tridiag = TriDiag(similar(Ps), similar(Ps), similar(Ps))

    k = Int(round(Nt / 1024))
    max_N2 = zeros(type, Int(round(Nt / k)))
    max_Pow = zeros(type, Int(round(Nt / k)))

    for t in 0:Nt-1

        # Update populations
        AtoPopUpdate!(N1, N2, Ps, CC1, dt1, λs)

        # Seed terms
        fs(u) = vg(λs) .* ((σe(λs) .* N2 .- σa(λs) .* N1) .* abs.(u) .- α .* abs.(u))
        seed.f = fs
        one_step!(seed, tridiag, t, dt1, Ps)

        if t % k == 0
            #p = plot(z2, Pp, lw=2, color="green", label="Pump", ylabel="Peak Power (W)", xlabel="Position (m)", legend= :topleft, ylims=(0, Pp0))
            #plot!(z, Ps ./ (Ps0), lw=2, color="red", label="Seed")
            #yaxis2 = twinx()
            #plot!(yaxis2, z2, N2 .+ 1e-12, lw=2, color="black", label="Excited Ions", ylabel="Population (1/m³)", ylims=(1,1e25), legend= :topright, yscale=:log10)
            #savefig("pumping_stage"*string(t))
            println(t, " steps taken")
            max_N2[Int(t / k) + 1] = maximum(N2)
            max_Pow[Int(t / k) + 1] = maximum(Pp)
            #ylims!(0, 1)
            #display(p)
        end

    end

end

println("DRE.jl compiled")
#using ProfileView
#ProfileView.@profview time_evolution(nsteps2, max_N2, max_Pow)
