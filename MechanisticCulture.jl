using Plots
using Distributions
using LaTeXStrings

function getQ(g)
    return collect(0:1/(2^g):1)
end
# =============================================================================
# Verdu 2011
# =============================================================================
function P_Verdu(s1_0, s2_0, s1_vec, s2_vec, tmax)
    Q_all = Vector{Vector{Float64}}()
    P_all = Vector{Vector{Float64}}()
    push!(Q_all, [0.0]); push!(P_all, [0.0])

    # g = 1
    push!(Q_all, getQ(1))
    push!(P_all, [s2_0^2, 2*s1_0*s2_0, s1_0^2])

    for g in 2:tmax+1
        Q_prev = Q_all[g]
        P_prev = P_all[g]
        s1g = s1_vec[g]
        s2g = s2_vec[g]
        hg  = 1.0 - s1g - s2g
        Qg  = getQ(g)
        Pg  = zeros(length(Qg))
        for idx_q in 1:length(Qg)
            q = Qg[idx_q]

            # HH
            hh_sum = 0.0
            for r in 0:2^(g-1)
                v1 = r / 2^(g-1)
                v2 = (2^g*q - r) / 2^(g-1)
                i1 = findfirst(v -> v == v1, Q_prev)
                i2 = findfirst(v -> v == v2, Q_prev)
                (i1 !== nothing && i2 !== nothing) && (hh_sum += P_prev[i1] * P_prev[i2])
            end

            # S1H
            i1 = findfirst(v -> v == 2*q - 1, Q_prev)
            s1h = (i1 !== nothing) ? 2*s1g*hg*P_prev[i1] : 0.0

            # S2H
            i2 = findfirst(v -> v == 2*q, Q_prev)
            s2h = (i2 !== nothing) ? 2*s2g*hg*P_prev[i2] : 0.0

            # I
            I_g = q==1.0 ? s1g^2 : q==0.5 ? 2*s1g*s2g : q==0.0 ? s2g^2 : 0.0

            Pg[idx_q] = hg^2*hh_sum + s1h + s2h + I_g
        end
        push!(Q_all, Qg); push!(P_all, Pg)
    end
    return Q_all, P_all
end
function Expectation_verdu(s1_0::Float64, s1::Vector{Float64}, s2::Vector{Float64}, tmax::Int)
    # Generation 0
    #E_h1[1] = 0.0   # H_{1,0} = 0
    E_h1 = zeros(Float64, tmax)
    # Generation 1 (Eq. 10)
    E_h1[1] = s1_0
    # Generations >= 2 (Eq. 11)
    for g in 2:tmax
        hprev = 1.0 - s1[g-1] - s2[g-1]
        E_h1[g] = s1[g-1] + hprev * E_h1[g-1]
    end
    return E_h1
end

# =============================================================================
# x' and y' weights for A (b weighted) and a ((1-b) weighted)
# =============================================================================
function get_xprime(s1, s2, h, x, b, θ)
    b3, b2, b1, b0 = b[:b3], b[:b2], b[:b1], b[:b0]

    AA = b3 * x^2 * 
            (s1^2 * (1+θ[:A11])^2 + s2^2 * (1+θ[:A22])^2 + h^2  * (1+θ[:AHH])^2 +
            2*h*s1 * (1+θ[:AH1])*(1+θ[:A1H]) +
            2*h*s2 * (1+θ[:AH2])*(1+θ[:A2H]) +
            2*s1*s2 * (1+θ[:A12])*(1+θ[:A21]))

    Aa = (b1 + b2) * x*(1-x) * 
            (s1^2 * (1+θ[:A11])*(1+θ[:a11]) + s2^2 * (1+θ[:A22])*(1+θ[:a22]) + h^2  * (1+θ[:AHH])*(1+θ[:aHH]) +
            h*s1 * ((1+θ[:AH1])*(1+θ[:a1H]) + (1+θ[:aH1])*(1+θ[:A1H])) +
            h*s2 * ((1+θ[:AH2])*(1+θ[:a2H]) + (1+θ[:aH2])*(1+θ[:A2H])) +
            s1*s2 * ((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])))

    aa = b0 * (1-x)^2 * 
            (s1^2 * (1+θ[:a11])^2 + s2^2 * (1+θ[:a22])^2 + h^2  * (1+θ[:aHH])^2 +
            2*h*s1 * (1+θ[:aH1])*(1+θ[:a1H]) +
            2*h*s2 * (1+θ[:aH2])*(1+θ[:a2H]) +
            2*s1*s2 * (1+θ[:a21])*(1+θ[:a12]))

    return AA + Aa + aa
end

function get_yprime(s1, s2, h, x, b, θ)
    b3, b2, b1, b0 = b[:b3], b[:b2], b[:b1], b[:b0]

    AA = (1-b3) * x^2 * ( s1^2 * (1+θ[:A11])^2 + s2^2 * (1+θ[:A22])^2 + h^2  * (1+θ[:AHH])^2 +
            2*h*s1 * (1+θ[:AH1])*(1+θ[:A1H]) +
            2*h*s2 * (1+θ[:AH2])*(1+θ[:A2H]) +
            2*s1*s2 * (1+θ[:A12])*(1+θ[:A21]))

    Aa = (2 - b1 - b2) * x*(1-x) * (s1^2 * (1+θ[:A11])*(1+θ[:a11]) + s2^2 * (1+θ[:A22])*(1+θ[:a22]) + h^2  * (1+θ[:AHH])*(1+θ[:aHH]) +
            h*s1 * ((1+θ[:AH1])*(1+θ[:a1H]) + (1+θ[:aH1])*(1+θ[:A1H])) +
            h*s2 * ((1+θ[:AH2])*(1+θ[:a2H]) + (1+θ[:aH2])*(1+θ[:A2H])) +
            s1*s2 * ((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])))

    aa = (1-b0) * (1-x)^2 * ( s1^2 * (1+θ[:a11])^2 + s2^2 * (1+θ[:a22])^2 + h^2  * (1+θ[:aHH])^2 +
            2*h*s1 * (1+θ[:aH1])*(1+θ[:a1H]) +
            2*h*s2 * (1+θ[:aH2])*(1+θ[:a2H]) +
            2*s1*s2 * (1+θ[:a21])*(1+θ[:a12]))

    return AA + Aa + aa
end
# =======================================================================================================
# Expectations 
# =======================================================================================================
function E_H_A1_gen1(s1_0, s2_0, x0_prime, x0, b, θ)
    term1 = s1_0^2 * F_S1S1_A(x0, b, θ)
    term2 = 0.5 * s1_0 * s2_0 * F_S1S2_A(x0, b, θ)
    return (term1 + term2) / x0_prime
end

function E_H_a1_gen1(s1_0, s2_0, y0_prime, x0, b, θ)
    term1 = s1_0^2 * F_S1S1_a(x0, b, θ)
    term2 = 0.5 * s1_0 * s2_0 * F_S1S2_a(x0, b, θ)
    return (term1 + term2) / y0_prime
end

function E_H_A1_next(E_prevA, E_prevа, s1, s2, h, x, x_prime, b, θ)
    # Constant part (from S1S1, S1S2, and the "1" in S1H expectation)
    const_part = (
        s1^2 * (
            b[:b3]*x^2*(1+θ[:A11])^2 +
            (b[:b1]+b[:b2])*x*(1-x)*(1+θ[:A11])*(1+θ[:a11]) +
            b[:b0]*(1-x)^2*(1+θ[:a11])^2
        ) +
        0.5*s1*s2 * (
            2*b[:b3]*x^2*(1+θ[:A12])*(1+θ[:A21]) +
            (b[:b1]+b[:b2])*x*(1-x)*((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])) +
            2*b[:b0]*(1-x)^2*(1+θ[:a12])*(1+θ[:a21])
        ) +
        0.5*s1*h * (
            2*b[:b3]*x^2*(1+θ[:A1H])*(1+θ[:AH1]) +
            (b[:b1]+b[:b2])*x*(1-x)*((1+θ[:AH1])*(1+θ[:a1H]) + (1+θ[:A1H])*(1+θ[:aH1])) +
            2*b[:b0]*(1-x)^2*(1+θ[:aH1])*(1+θ[:a1H])
        )
    ) / x_prime


    # r^A_k terms
    rA1 = b[:b3] * x^2
    rA2 = (b[:b1] + b[:b2]) * x * (1-x)
    rA3 = b[:b0] * (1-x)^2

    # W^{A,A}_{1H,g} and W^{A,a}_{1H,g}
    W1H_AA = 2*rA1*(1+θ[:A1H])*(1+θ[:AH1]) +   rA2*(1+θ[:AH1])*(1+θ[:a1H])
    W1H_Aa =   rA2*(1+θ[:A1H])*(1+θ[:aH1]) + 2*rA3*(1+θ[:a1H])*(1+θ[:aH1])

    # W^{A,A}_{2H,g} and W^{A,a}_{2H,g}
    W2H_AA = 2*rA1*(1+θ[:A2H])*(1+θ[:AH2]) +   rA2*(1+θ[:AH2])*(1+θ[:a2H])
    W2H_Aa =   rA2*(1+θ[:A2H])*(1+θ[:aH2]) + 2*rA3*(1+θ[:a2H])*(1+θ[:aH2])

    # AHH×aHH cross term 
    hh_cross = 0.5 * rA2 * (1+θ[:AHH]) * (1+θ[:aHH])

    # Coefficient of E[H_{A1,g-1}]:
    coef_A = (
        0.5*s1*h * W1H_AA +
        0.5*s2*h * W2H_AA +
        h^2 * ( rA1*(1+θ[:AHH])^2 + hh_cross )
    ) / x_prime

    # Coefficient of E[H_{a1,g-1}]:
    coef_a = (
        0.5*s1*h * W1H_Aa +
        0.5*s2*h * W2H_Aa +
        h^2 * ( rA3*(1+θ[:aHH])^2 + hh_cross )
    ) / x_prime

    return const_part + coef_A * E_prevA + coef_a * E_prevа
end

function E_H_a1_next(E_prevA, E_prevа, s1, s2, h, x, y_prime, b, θ)
    # Constant part
    const_part = (
        s1^2 * (
            (1-b[:b3])*x^2*(1+θ[:A11])^2 +
            (2-b[:b1]-b[:b2])*x*(1-x)*(1+θ[:A11])*(1+θ[:a11]) +
            (1-b[:b0])*(1-x)^2*(1+θ[:a11])^2
        ) +
        0.5*s1*s2 * (
            2*(1-b[:b3])*x^2*(1+θ[:A12])*(1+θ[:A21]) +
            (2-b[:b1]-b[:b2])*x*(1-x)*((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])) +
            2*(1-b[:b0])*(1-x)^2*(1+θ[:a12])*(1+θ[:a21])
        ) +
        0.5*s1*h * (
            2*(1-b[:b3])*x^2*(1+θ[:A1H])*(1+θ[:AH1]) +
            (2-b[:b1]-b[:b2])*x*(1-x)*((1+θ[:AH1])*(1+θ[:a1H]) + (1+θ[:A1H])*(1+θ[:aH1])) +
            2*(1-b[:b0])*(1-x)^2*(1+θ[:aH1])*(1+θ[:a1H])
        )
    ) / y_prime
    # r^a_k terms
    ra1 = (1 - b[:b3]) * x^2
    ra2 = (2 - b[:b1] - b[:b2]) * x * (1-x)
    ra3 = (1 - b[:b0]) * (1-x)^2

    # W^{a,A}_{1H,g} and W^{a,a}_{1H,g}
    W1H_aA = 2*ra1*(1+θ[:A1H])*(1+θ[:AH1]) +   ra2*(1+θ[:AH1])*(1+θ[:a1H])
    W1H_aa =   ra2*(1+θ[:A1H])*(1+θ[:aH1]) + 2*ra3*(1+θ[:a1H])*(1+θ[:aH1])

    # W^{a,A}_{2H,g} and W^{a,a}_{2H,g}
    W2H_aA = 2*ra1*(1+θ[:A2H])*(1+θ[:AH2]) +   ra2*(1+θ[:AH2])*(1+θ[:a2H])
    W2H_aa =   ra2*(1+θ[:A2H])*(1+θ[:aH2]) + 2*ra3*(1+θ[:a2H])*(1+θ[:aH2])

    hh_cross = 0.5 * ra2 * (1+θ[:AHH]) * (1+θ[:aHH])

    coef_A = (
        0.5*s1*h * W1H_aA +
        0.5*s2*h * W2H_aA +
        h^2 * ( ra1*(1+θ[:AHH])^2 + hh_cross )
    ) / y_prime

    coef_a = (
        0.5*s1*h * W1H_aa +
        0.5*s2*h * W2H_aa +
        h^2 * ( ra3*(1+θ[:aHH])^2 + hh_cross )
    ) / y_prime

    return const_part + coef_A * E_prevA + coef_a * E_prevа
end

# =============================================================================
# Variance E[H_A^2] and E[H_a^2]
# =============================================================================
function E_H2_A1_gen1(s1_0, s2_0, xp, x0, b, θ) # Equation 26 for A 
    # S1S1 q^2=1  S1S2 q^2=1/4  S2S2  q=0
    term1 = s1_0^2 * F_S1S1_A(x0, b, θ) / xp          # * 1 F_S1S1_A corresponds to W\bullet11 in manuscript
    term2 = 0.25 * s1_0 * s2_0 * F_S1S2_A(x0, b, θ) / xp  # * 1/4
    return term1 + term2
end

function E_H2_a1_gen1(s1_0, s2_0, yp, x0, b, θ) # Equation 26 for a
    term1 = s1_0^2 * F_S1S1_a(x0, b, θ) / yp
    term2 = 0.25 * s1_0 * s2_0 * F_S1S2_a(x0, b, θ) / yp
    return term1 + term2
end

function E_H2_A1_next(E2_prevA, E2_preva, E_prevA, E_preva,
                      s1, s2, h, x, x_prime, b, θ)
    # r^A_k terms
    rA1 = b[:b3] * x^2
    rA2 = (b[:b1] + b[:b2]) * x * (1-x)
    rA3 = b[:b0] * (1-x)^2

    # W^{A,A}_{1H,g} and W^{A,a}_{1H,g}
    W1H_AA = 2*rA1*(1+θ[:A1H])*(1+θ[:AH1]) +   rA2*(1+θ[:AH1])*(1+θ[:a1H])
    W1H_Aa =   rA2*(1+θ[:A1H])*(1+θ[:aH1]) + 2*rA3*(1+θ[:a1H])*(1+θ[:aH1])

    # W^{A,A}_{2H,g} and W^{A,a}_{2H,g}
    W2H_AA = 2*rA1*(1+θ[:A2H])*(1+θ[:AH2]) +   rA2*(1+θ[:AH2])*(1+θ[:a2H])
    W2H_Aa =   rA2*(1+θ[:A2H])*(1+θ[:aH2]) + 2*rA3*(1+θ[:a2H])*(1+θ[:aH2])

    # Constant part
    const_part = (s1^2 * (rA1*(1+θ[:A11])^2 + rA2*(1+θ[:A11])*(1+θ[:a11]) + rA3*(1+θ[:a11])^2) +
                  0.25 * s1*s2 * (2*rA1*(1+θ[:A12])*(1+θ[:A21]) +
                                 rA2*((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])) +
                                 2*rA3*(1+θ[:a12])*(1+θ[:a21]))) / x_prime

    # S1H term
    s1h_part =  s1 * h * (W1H_AA * (1 + 2*E_prevA + E2_prevA) / 4 +
                          W1H_Aa * (1 + 2*E_preva + E2_preva) / 4) / x_prime

    # S2H term
    s2h_part = s2 * h * (W2H_AA * E2_prevA / 4 +
                         W2H_Aa * E2_preva / 4 ) / x_prime 

    # HH term
    hh_AA = (E2_prevA + E_prevA^2) / 2
    hh_aa = (E2_preva + E_preva^2) / 2
    hh_Aa = (E2_prevA + E2_preva + 2*E_prevA*E_preva) / 4
    hh_part = h^2 * (rA1 * (1+θ[:AHH])^2 * hh_AA +
                     rA2 * (1+θ[:AHH])*(1+θ[:aHH]) * hh_Aa +
                     rA3 * (1+θ[:aHH])^2 * hh_aa) / x_prime
    return const_part + s1h_part + s2h_part + hh_part
end

function E_H2_a1_next(E2_prevA, E2_preva, E_prevA, E_preva,
                      s1, s2, h, x, y_prime, b, θ)
    # r^a_k terms
    ra1 = (1 - b[:b3]) * x^2
    ra2 = (2 - b[:b1] - b[:b2]) * x * (1-x)
    ra3 = (1 - b[:b0]) * (1-x)^2

    # W^{a,A}_{1H,g} and W^{a,a}_{1H,g}
    W1H_aA = 2*ra1*(1+θ[:A1H])*(1+θ[:AH1]) +   ra2*(1+θ[:AH1])*(1+θ[:a1H])
    W1H_aa =   ra2*(1+θ[:A1H])*(1+θ[:aH1]) + 2*ra3*(1+θ[:a1H])*(1+θ[:aH1])

    # W^{a,A}_{2H,g} and W^{a,a}_{2H,g}
    W2H_aA = 2*ra1*(1+θ[:A2H])*(1+θ[:AH2]) +   ra2*(1+θ[:AH2])*(1+θ[:a2H])
    W2H_aa =   ra2*(1+θ[:A2H])*(1+θ[:aH2]) + 2*ra3*(1+θ[:a2H])*(1+θ[:aH2])

    # Constant part
    const_part = (s1^2 * (ra1*(1+θ[:A11])^2 +
                          ra2*(1+θ[:A11])*(1+θ[:a11]) +
                          ra3*(1+θ[:a11])^2) +
                  0.25*s1*s2 * (2*ra1*(1+θ[:A12])*(1+θ[:A21]) +
                                ra2*((1+θ[:A12])*(1+θ[:a21]) + (1+θ[:A21])*(1+θ[:a12])) +
                                2*ra3*(1+θ[:a12])*(1+θ[:a21]))) / y_prime

    # S1H term
    s1h_part = s1 * h * (W1H_aA * (1 + 2*E_prevA + E2_prevA) / 4 +
                         W1H_aa * (1 + 2*E_preva + E2_preva) / 4)/ y_prime

    # S2H term
    s2h_part = s2 * h * (W2H_aA * E2_prevA / 4 +
                         W2H_aa * E2_preva / 4) / y_prime

    # HH term
    hh_AA = (E2_prevA + E_prevA^2) / 2
    hh_aa = (E2_preva + E_preva^2) / 2
    hh_Aa = (E2_prevA + E2_preva + 2*E_prevA*E_preva) / 4
    hh_part = h^2 * (ra1 * (1+θ[:AHH])^2 * hh_AA +
                     ra2 * (1+θ[:AHH])*(1+θ[:aHH]) * hh_Aa +
                     ra3 * (1+θ[:aHH])^2 * hh_aa) / y_prime
    return const_part + s1h_part + s2h_part + hh_part
end

function Var_H_bullet_gen1(s1_0, s2_0, xp, yp, b, θ) # Equation 27 V[H_{\bullet 1,1}]
    # A carriers
    E_HA_sq = E_H2_A1_gen1(s1_0, s2_0, xp, x0, b, θ)
    E_HA    = E_H_A1_gen1(s1_0, s2_0, xp, x0, b, θ)
    Var_HA  = E_HA_sq - E_HA^2

    # a carriers
    E_Ha_sq = E_H2_a1_gen1(s1_0, s2_0, yp, x0, b, θ)
    E_Ha    = E_H_a1_gen1(s1_0, s2_0, yp, x0, b, θ)
    Var_Ha  = E_Ha_sq - E_Ha^2

    # Total variance
    x1 = xp / (xp + yp)
    Var_HT = x1 * Var_HA + (1 - x1) * Var_Ha #+ x1 * (1 - x1) * (E_HA - E_Ha)^2

    return Var_HA, Var_Ha, Var_HT
end

# next generations (g >= 2)
function Var_H_bullet_next(g, E2_HA_prev, E2_Ha_prev, E_HA_prev, E_Ha_prev,
                           s1, s2, h, x, xp, yp, b, θ)
    # A carriers
    E_HA_sq = E_H2_A1_next(E2_HA_prev, E2_Ha_prev, E_HA_prev, E_Ha_prev,
                           s1, s2, h, x, xp, b, θ)
    E_HA    = E_H_A1_next(E_HA_prev, E_Ha_prev, s1, s2, h, x, xp, b, θ)
    Var_HA  = E_HA_sq - E_HA^2

    # a carriers
    E_Ha_sq = E_H2_a1_next(E2_HA_prev, E2_Ha_prev, E_HA_prev, E_Ha_prev,
                           s1, s2, h, x, yp, b, θ)
    E_Ha    = E_H_a1_next(E_HA_prev, E_Ha_prev, s1, s2, h, x, yp, b, θ)
    Var_Ha  = E_Ha_sq - E_Ha^2

    # Total variance
    xg = xp / (xp + yp)
    Var_HT = xg * Var_HA + (1 - xg) * Var_Ha + xg * (1 - xg) * (E_HA - E_Ha)^2

    return Var_HA, Var_Ha, Var_HT
end


# =============================================================================
# Weighted mating frequency functions
# =============================================================================
# S1 S1 
F_S1S1_A(x, b, θ) = b[:b3] * x^2 * (1+θ[:A11])^2 +
                     (b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:A11]) * (1+θ[:a11]) +
                     b[:b0] * (1-x)^2 * (1+θ[:a11])^2

F_S1S1_a(x, b, θ) = (1-b[:b3]) * x^2 * (1+θ[:A11])^2 +
                     (2 - b[:b1] - b[:b2]) * x * (1-x) * (1+θ[:A11]) * (1+θ[:a11]) +
                     (1 - b[:b0]) * (1-x)^2 * (1+θ[:a11])^2

# S1 S2 
F_S1S2_A(x, b, θ) = 2 * b[:b3] * x^2 * (1+θ[:A12])*(1+θ[:A21]) +
                     (b[:b1]+b[:b2]) * x * (1-x) * ((1+θ[:A12]) * (1+θ[:a21]) + (1+θ[:A21]) * (1+θ[:a12])) +
                    2 *b[:b0] * (1-x)^2 * (1+θ[:a12]) * (1+θ[:a21])
 
F_S1S2_a(x, b, θ) = 2 * (1-b[:b3]) * x^2 * (1+θ[:A12]) * (1+θ[:A21]) +
                    (2 - b[:b1] - b[:b2]) * x * (1-x) * ((1+θ[:A12]) * (1+θ[:a21]) + (1+θ[:A21]) * (1+θ[:a12])) +
                    2 * (1-b[:b0]) * (1-x)^2 * (1+θ[:a12]) * (1+θ[:a21])

# S2 S2 
F_S2S2_A(x, b, θ) = b[:b3]*x^2*(1+θ[:A22])^2 +
                     (b[:b1]+b[:b2])*x*(1-x)*(1+θ[:A22])*(1+θ[:a22]) +
                     b[:b0]*(1-x)^2*(1+θ[:a22])^2

F_S2S2_a(x, b, θ) = (1-b[:b3])*x^2*(1+θ[:A22])^2 +
                     (2-b[:b1]-b[:b2])*x*(1-x)*(1+θ[:A22])*(1+θ[:a22]) +
                     (1-b[:b0])*(1-x)^2*(1+θ[:a22])^2



# =============================================================================
# g = 1; only S1S1, S1S2, S2S2 possible (no H parents yet)
# in text P(H_A1,1 = q) and P(H_a1,1 = q)
# =============================================================================
function compute_gen1(s1, s2, x, b, θ)
    xp = get_xprime(s1, s2, 0.0, x, b, θ)   # h=0 at g=0
    yp = get_yprime(s1, s2, 0.0, x, b, θ)

    Q1 = getQ(1)   # [0, 0.5, 1]
    P1_A = zeros(3)
    P1_a = zeros(3)

    # q = 0  S2S2, H_A = 0
    P1_A[1] = s2^2 * F_S2S2_A(x, b, θ) / xp
    P1_a[1] = s2^2 * F_S2S2_a(x, b, θ) / yp

    # q = 1/2 S1S2, H_A = 1/2
    P1_A[2] = s1 * s2 * F_S1S2_A(x, b, θ) / xp
    P1_a[2] = s1 * s2 * F_S1S2_a(x, b, θ) / yp

    # q = 1  S1S1, H_A = 1
    P1_A[3] = s1^2 * F_S1S1_A(x, b, θ) / xp
    P1_a[3] = s1^2 * F_S1S1_a(x, b, θ) / yp

    return Q1, P1_A, P1_a, xp, yp
end
# =====================================================================================
# g >= 2; basically P(H_A1,g = q) and P(H_a1,g = q)
# I_A,g(q): S1S1, S1S2, and S2S2 offspring
# S1H term: S1H  P(H = 2q-1)
# S2H term: S2H  P(H = 2q)
# HH term: HH  iterate over r
# =====================================================================================
function compute_gen_g(g, Q_prev, P_prevA, P_preva, s1, s2, h, x, b, θ)
    xp = get_xprime(s1, s2, h, x, b, θ)
    yp = get_yprime(s1, s2, h, x, b, θ)
    Qg  = getQ(g)
    PgA = zeros(length(Qg))
    Pga = zeros(length(Qg))

    for idx_q in 1:length(Qg)
        q = Qg[idx_q]
        # ------------------------------------------------------------------
        # I term 
        # ------------------------------------------------------------------
        I_A = 0.0
        I_a = 0.0
        if q == 1.0
            I_A = s1^2 * F_S1S1_A(x, b, θ) / xp
            I_a = s1^2 * F_S1S1_a(x, b, θ) / yp
        elseif q == 0.5
            I_A = s1 * s2 * F_S1S2_A(x, b, θ) / xp
            I_a = s1 * s2 * F_S1S2_a(x, b, θ) / yp
        elseif q == 0.0
            I_A = s2^2 * F_S2S2_A(x, b, θ) / xp
            I_a = s2^2 * F_S2S2_a(x, b, θ) / yp
        end

        # ------------------------------------------------------------------
        # S1H offspring inherits fraction 1 from S1 parent, H_{1,g-1} from H parent
        # hybrid fraction = (1 + H_{1,g-1}) / 2, so we need H_{1,g-1} = 2q - 1
        # ------------------------------------------------------------------
        term_S1HA_A = 0.0
        term_S1Ha_A = 0.0
        term_S1HA_a = 0.0
        term_S1Ha_a = 0.0
        S1H_A_term  = 0.0     
        S1H_a_term  = 0.0 
        idx_2q1 = findfirst(v -> v == 2*q - 1, Q_prev)
        if idx_2q1 !== nothing
            PHA = P_prevA[idx_2q1]
            PHa = P_preva[idx_2q1]
            term_S1HA_A = (2 * b[:b3] * x^2 * (1+θ[:A1H]) * (1+θ[:AH1]) +
                          (b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:AH1]) * (1+θ[:a1H])) * PHA 

            term_S1Ha_A = ((b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:A1H]) * (1+θ[:aH1]) +
                          2 * b[:b0] * (1-x)^2 * (1+θ[:aH1]) * (1+θ[:a1H])) * PHa
           
            term_S1HA_a = (2 * (1-b[:b3]) * x^2 * (1+θ[:A1H]) * (1+θ[:AH1]) +
                          (2-b[:b1]-b[:b2]) * x * (1-x) * (1+θ[:AH1]) * (1+θ[:a1H])) * PHA


            term_S1Ha_a = ((2-b[:b1]-b[:b2]) * x * (1-x) * (1+θ[:A1H]) * (1+θ[:aH1]) +
                          2 * (1-b[:b0]) * (1-x)^2 * (1+θ[:aH1]) * (1+θ[:a1H])) * PHa

            S1H_A_term = ((s1 * h) / xp) * (term_S1HA_A + term_S1Ha_A)
            S1H_a_term = ((s1 * h) / yp) * (term_S1HA_a + term_S1Ha_a)
        end

        # ------------------------------------------------------------------
        # S2H offspring inherits 0 from S2 parent, H_{1,g-1} from H parent
        # hybrid fraction = H_{1,g-1} / 2, so we need H_{1,g-1} = 2q
        # ------------------------------------------------------------------
        term_S2HA_A = 0.0
        term_S2Ha_A = 0.0
        term_S2HA_a = 0.0
        term_S2Ha_a = 0.0
        S2H_A_term = 0.0
        S2H_a_term = 0.0
        idx_2q = findfirst(v -> v == 2*q, Q_prev)
        if idx_2q !== nothing
            PHA = P_prevA[idx_2q]
            PHa = P_preva[idx_2q]
            term_S2HA_A = (2 * b[:b3] * x^2 * (1+θ[:A2H]) * (1+θ[:AH2]) +
                          (b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:AH2]) * (1+θ[:a2H])) * PHA

            term_S2Ha_A = ((b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:A2H]) * (1+θ[:aH2]) +
                          2 * b[:b0] * (1-x)^2 * (1+θ[:aH2]) * (1+θ[:a2H])) * PHa 

            term_S2HA_a = (2 * (1-b[:b3]) * x^2 * (1+θ[:A2H]) * (1+θ[:AH2]) +
                          (2-b[:b1]-b[:b2]) * x * (1-x) * (1+θ[:AH2]) * (1+θ[:a2H])) * PHA

            term_S2Ha_a = ((2-b[:b1]-b[:b2]) * x * (1-x) * (1+θ[:A2H]) * (1+θ[:aH2]) +
                          2 * (1-b[:b0]) * (1-x)^2 * (1+θ[:aH2]) * (1+θ[:a2H])) * PHa


            S2H_A_term = ((s2 * h) / xp) * (term_S2HA_A + term_S2Ha_A)
            S2H_a_term = ((s2 * h) / yp) * (term_S2HA_a + term_S2Ha_a)
        end

        # ------------------------------------------------------------------
        # HH term hybrid fraction = (H_{1,g-1}^(1) + H_{1,g-1}^(2)) / 2
        # sum over r = 0..2^(g-1) such that r/2^(g-1) + (2^g*q - r)/2^(g-1) = 2q
        # account for AA, Aa , aa
        # ------------------------------------------------------------------
        sum_AA = 0.0
        sum_Aa = 0.0
        sum_aa = 0.0
        
        for r in 0:2^(g-1)
            val1 = r / 2^(g-1)
            val2 = (2^g * q - r) / 2^(g-1)
            i1 = findfirst(v -> v == val1, Q_prev)
            i2 = findfirst(v -> v == val2, Q_prev)
            if i1 !== nothing && i2 !== nothing
                sum_AA += P_prevA[i1] * P_prevA[i2]
                sum_Aa += P_prevA[i1] * P_preva[i2]  
                sum_aa += P_preva[i1] * P_preva[i2]
            end
        end
        term_AA_A = (b[:b3] * x^2 * (1+θ[:AHH])^2) * sum_AA
        term_Aa_A = ((b[:b1]+b[:b2]) * x * (1-x) * (1+θ[:AHH]) * (1+θ[:aHH])) * sum_Aa
        term_aa_A = (b[:b0] * (1-x)^2 * (1+θ[:aHH])^2) * sum_aa

        term_AA_a = ((1-b[:b3])*x^2*(1+θ[:AHH])^2) * sum_AA
        term_Aa_a = ((2-b[:b1]-b[:b2])*x*(1-x)*(1+θ[:AHH])*(1+θ[:aHH])) * sum_Aa
        term_aa_a = ((1-b[:b0])*(1-x)^2*(1+θ[:aHH])^2) * sum_aa

        term_HH_A = (h^2 / xp) * (term_AA_A + term_Aa_A + term_aa_A)
        term_HH_a = (h^2 / yp) * (term_AA_a + term_Aa_a + term_aa_a)
        # -----
        # Sum
        # -----
        PgA[idx_q] = I_A + S1H_A_term + S2H_A_term + term_HH_A
        Pga[idx_q] = I_a + S1H_a_term + S2H_a_term + term_HH_a
    end

    return Qg, PgA, Pga, xp, yp
end
# =============================================================================
# Master runner: given b and θ, run tmax generations and return
# E[H_T,g] and Var[H_T,g] for g = 1..tmax
# =============================================================================
function run_model(b, θ, s1_vec, s2_vec, x0, tmax)
    Q_all    = Vector{Vector{Float64}}()
    P_all_HA = Vector{Vector{Float64}}()
    P_all_Ha = Vector{Vector{Float64}}()
    E_HA = zeros(tmax + 1)
    E_Ha = zeros(tmax + 1)
    E_HT = zeros(tmax + 1) # E[H_{1,g}]

    E2_HA = zeros(tmax + 1)   # E[H_A^2]
    E2_Ha = zeros(tmax + 1)   # E[H_a^2]
    Var_HA = zeros(tmax + 1)
    Var_Ha = zeros(tmax + 1)
    Var_HT = zeros(tmax + 1)


    # --- g = 0 ---
    push!(Q_all,   [0.0])
    push!(P_all_HA,[0.0])
    push!(P_all_Ha,[0.0])
    x_vec    = Float64[x0]
    h_vec    = Float64[1.0 - s1_vec[1] - s2_vec[1]]

     E_HA[1] = 0.0
     E_Ha[1] = 0.0
     E_HT[1] = 0.0

    E2_HA[1] = 0
    E2_Ha[1] = 0

    # --- g = 1 ---
    Q1, P1_A, P1_a, xp1, yp1 = compute_gen1(s1_vec[1], s2_vec[1], x0, b, θ) 
    push!(Q_all,    Q1)
    push!(P_all_HA, P1_A)
    push!(P_all_Ha, P1_a)
    push!(x_vec, xp1 / (xp1 + yp1))
    push!(h_vec, 1.0 - s1_vec[2] - s2_vec[2])
    # calculate expectation at g = 1
     E_HA[2] = E_H_A1_gen1(s1_vec[1], s2_vec[1], xp1, x0, b, θ)
     E_Ha[2] = E_H_a1_gen1(s1_vec[1], s2_vec[1], yp1, x0, b, θ)
     E_H_total = x_vec[2] * E_HA[2] + (1 - x_vec[2]) * E_Ha[2]
     E_HT[2] = E_H_total

    # calculate variance at g = 1
    E2_HA[2] = E_H2_A1_gen1(s1_vec[1], s2_vec[1], xp1, x0, b, θ)
    E2_Ha[2] = E_H2_a1_gen1(s1_vec[1], s2_vec[1], yp1, x0, b, θ)
    Var_HA[2], Var_Ha[2], Var_HT[2] = Var_H_bullet_gen1(s1_vec[1], s2_vec[1], xp1, yp1, b, θ)


    # --- g >= 2 to tmax ---
    for g in 2:tmax
    # parameters at generation g-1 (prev)
    s1_prev = s1_vec[g]
    s2_prev = s2_vec[g]
    h_prev  = h_vec[g]
    x_prev  = x_vec[g]

    Qg, PgA, Pga, xp, yp = compute_gen_g(g, Q_all[end], P_all_HA[end], P_all_Ha[end],s1_prev, s2_prev, h_prev, x_prev, b, θ)
    push!(Q_all, Qg)
    push!(P_all_HA, PgA)
    push!(P_all_Ha, Pga)
    push!(x_vec, xp / (xp + yp))
    push!(h_vec, 1.0 - s1_vec[g+1] - s2_vec[g+1])

    # expectation calculation
     E_HA[g+1] = E_H_A1_next(E_HA[g], E_Ha[g], s1_prev, s2_prev, h_prev, x_prev, xp, b, θ)
     E_Ha[g+1] = E_H_a1_next(E_HA[g], E_Ha[g], s1_prev, s2_prev, h_prev, x_prev, yp, b, θ)
     E_H_total = x_vec[g+1] * E_HA[g+1] + (1 - x_vec[g+1]) * E_Ha[g+1]   # E_HT[g]
     E_HT[g+1] = E_H_total


    # variance calculation
    E2_HA[g+1] = E_H2_A1_next(E2_HA[g], E2_Ha[g], E_HA[g], E_Ha[g], s1_prev, s2_prev, h_prev, x_prev, xp, b, θ)
    E2_Ha[g+1] = E_H2_a1_next(E2_HA[g], E2_Ha[g], E_HA[g], E_Ha[g], s1_prev, s2_prev, h_prev, x_prev, yp, b, θ)
    Var_HA[g+1], Var_Ha[g+1], Var_HT[g+1] = Var_H_bullet_next(g, E2_HA[g], E2_Ha[g], E_HA[g], E_Ha[g], s1_prev, s2_prev, h_prev, x_prev, xp, yp, b, θ)    
    end

    return x_vec, E_HA, E_Ha, E_HT, E2_HA, E2_Ha, Var_HA, Var_Ha, Var_HT, Q_all, P_all_HA, P_all_Ha #x_vec, E_HA, E_Ha, E_HT, Q_all, P_all_HA, P_all_Ha
            
end

# =============================================================================
# Neutral example run
# =============================================================================
tmax   = 6
s1_vec = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
s2_vec = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
x0     = 0.5
θ_neutral = Dict(:A11 => 0.0, :A12 => 0.0, :A1H => 0.0,
                 :A21 => 0.0, :A22 => 0.0, :A2H => 0.0,
                 :AH1 => 0.0, :AH2 => 0.0, :AHH => 0.0,
                 :a11 => 0.0, :a12 => 0.0, :a1H => 0.0,
                 :a21 => 0.0, :a22 => 0.0, :a2H => 0.0,
                 :aH1 => 0.0, :aH2 => 0.0, :aHH => 0.0)
b = Dict(:b0 => 0.0, :b1 => 0.9, :b2 => 0.9, :b3 => 1.0)
x_vec, E_HA, E_Ha, E_HT, E2_HA, E2_Ha, Var_HA, Var_Ha, Var_HT, Q_all, P_all_HA, P_all_Ha = run_model(b, θ_neutral, s1_vec, s2_vec, x0, tmax)
