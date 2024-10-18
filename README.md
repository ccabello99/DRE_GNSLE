# RK4IP_DRE
Julia code that simulates the Dynamic Rate Equations (DRE) for a pump or seed pulse using the Crank-Nicolson method. Additionally, it can solve the Generalized Nonlinear Schrodinger Equation (GNSLE) to simulate nonlinear propagation with the Runge-Kutta 4 in the Interaction Picture (RK4IP) method, including in the presence of an amplifying medium. In the presence of amplification, the DRE & GNSLE will be solved simulataneously. The formulation accounts for strongly-chirped regime by calculating instantaneous wavelength in the amplifying medium. 

In order to use:
1) Compile GNSLE.jl
2) Run :
```julia
    DRE(Pump(), sim_params, 1)
```
 - to generate an excited population for one pump pass. This will create "N2_1_RK4.csv".
3) Run :
```julia
    DRE(Pump(), sim_params, 2)
```
 - to generate an excited population for two pump passes. It will use "N2_1_RK4.csv" from simulation for one pump pass and create a new "N2_1_RK4.csv".
4) Run :
```julia
    RK4IP(sim_params, pass=<pass_number>, pumped=<true/false>, visualize=<true/false>)
```
- If pumped = true and pass = 1, it will use "N2_1_RK4.csv" from pumping stage for initial population in excited state. If pumped=true and pass > 1, it will use populations saved after previous pass.
