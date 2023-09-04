### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 485f5fd6-44e3-11ee-2843-41acdf8b176c
let
    import Pkg
	cd(".")
    Pkg.activate(".")
end;

# ╔═╡ 358e224d-cdbf-4cb9-810b-a575041c125c
begin
	using Turing
	using Plots, StatsPlots
	using Flux, CUDA
	using CSV, DataFrames
end

# ╔═╡ f2f91f85-8085-463d-ac08-1177e951a7a1
md"""
# Temp
"""

# ╔═╡ aa33b828-730e-41fa-afcb-ad40206f2cdd
begin
	salary = 1
	salary_rate = 0.03
	interest_rate = 0.04
end;

# ╔═╡ c74ee6f6-fc39-478b-90f2-a75e60e38570
function PUAL(P, S, e, i, R, x, a, A)::Float64
	return P*S*((1+e)/(1+i))^(R-x)*a/A
end

# ╔═╡ 44269779-c645-4068-889a-65138da16ac9
function PUAL(past_service, x)::Float64
	return PUAL(past_service, salary, salary_rate, interest_rate, 65, x, 13.666, 1/65)
end

# ╔═╡ cd6384d1-3539-4fe9-9863-ef439d26b551
plot(25:65, PUAL.(0:40, 25:65), xlabel="Age", ylabel="PUAL", label="PUAL")

# ╔═╡ d2a8465b-4249-4e31-9097-fba91c4397ca
function R(x, t)::Float64
	α = 0.0
	f = 0.0
	if x < 60
		α = 0.13
		f = 0.55
	elseif 60 ≤ x ≤ 110
		α = 1-0.87*(110-x)/50
		f = 0.55*(110-x)/50 + 0.29*(x-60)/50
	else
		α = 1
		f = 0.29
	end
	return α + (1-α)*(1-f)^((t-1992)/20)
end

# ╔═╡ 557d73b0-48da-487a-9a4d-8fed1753d03f
function mortality_projection(qx, x, t)::Float64
	@assert t ≥ 1992
	return qx * R(x, t)
end

# ╔═╡ 343efad5-4b5a-4ec4-95af-c3622c0fb4c0
begin
	α_(x) = 1-0.67*(110-x)/50
	f_(x) = 0.55*(110-x)/50 + 0.29*(x-60)/50
	p1 = plot(60:100, α_.(60:100), xlabel="Age", ylabel="α(x)", label="")
	p2 = plot(60:100, f_.(60:100), xlabel="Age", ylabel="f(x)", label="")
	p3 = begin
			plot(xlabel="Age", ylabel="R(x, t)", label="", ylim=(0, 1))
			plot!(60:100, R.(60:100, 1992), label="1992")
			plot!(60:100, R.(60:100, 2002), label="2002")
			plot!(60:100, R.(60:100, 2012), label="2012")
			plot!(60:100, R.(60:100, 2022), label="2022")
			plot!(60:100, R.(60:100, 2032), label="2032")
		end
	plot(p1, p2, p3)
end

# ╔═╡ 2f0ef480-8cbb-41b6-97dc-fc09b0723320
begin
	plot(xlabel = "Year", ylabel = "Projected Mortality (age 65)")
	plot!(1992:2050, mortality_projection.(0.012211, 65, 1992:2050), label="PMA92Base")
	plot!(1992:2050, mortality_projection.(0.006032, 65, 1992:2050), label="PMA92C20")
end

# ╔═╡ Cell order:
# ╟─f2f91f85-8085-463d-ac08-1177e951a7a1
# ╟─485f5fd6-44e3-11ee-2843-41acdf8b176c
# ╠═358e224d-cdbf-4cb9-810b-a575041c125c
# ╠═aa33b828-730e-41fa-afcb-ad40206f2cdd
# ╠═c74ee6f6-fc39-478b-90f2-a75e60e38570
# ╠═44269779-c645-4068-889a-65138da16ac9
# ╠═cd6384d1-3539-4fe9-9863-ef439d26b551
# ╠═d2a8465b-4249-4e31-9097-fba91c4397ca
# ╠═557d73b0-48da-487a-9a4d-8fed1753d03f
# ╠═343efad5-4b5a-4ec4-95af-c3622c0fb4c0
# ╠═2f0ef480-8cbb-41b6-97dc-fc09b0723320
