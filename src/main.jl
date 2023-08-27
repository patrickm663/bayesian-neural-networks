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

# ╔═╡ c74ee6f6-fc39-478b-90f2-a75e60e38570


# ╔═╡ Cell order:
# ╟─f2f91f85-8085-463d-ac08-1177e951a7a1
# ╟─485f5fd6-44e3-11ee-2843-41acdf8b176c
# ╠═358e224d-cdbf-4cb9-810b-a575041c125c
# ╠═c74ee6f6-fc39-478b-90f2-a75e60e38570
