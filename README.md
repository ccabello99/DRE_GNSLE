# RK4IP_DRE
Julia code that simulates the Dynamic Rate Equations (DRE) for a pump or seed pulse using the Crank-Nicolson method. Additionally, it can solve the Generalized Nonlinear Schrodinger Equation (GNSLE) to simulate nonlinear propagation with the Runge-Kutta 4 in the Interaction Picture (RK4IP) method, including in the presence of an amplifying medium. In the presence of amplification, the DRE & GNSLE will be solved simulataneously.

In order to use:
1) Compile GNSLE.jl
2) Run :
```julia
    RK4IP(sim_params, pass=<pass_number>)
```
