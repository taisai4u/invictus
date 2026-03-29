from sympy import symbols, Rational, Float

P, P0, h = symbols("P P0 h")
exp = Rational(1903, 10000)

# Rearrange manually: h = 44330*(1 - (P/P0)^exp)
# => (P/P0)^exp = 1 - h/44330
# => P = P0 * (1 - h/44330)^(1/exp)

P_solution = P0 * (1 - h / 44330) ** (1 / exp)
P_derivative = P_solution.diff(h)

print(P_solution)
print(P_derivative)
