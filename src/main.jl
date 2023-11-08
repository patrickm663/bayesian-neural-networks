### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 0b0bb0d0-7d73-11ee-21ff-79bf038dbc5b
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
end

# ╔═╡ ee32c415-f6c5-49cd-97de-0213f01d0ec6
begin
	using PlutoUI
	using Flux, Random, MLDataUtils
	using CSV, DataFrames, LinearAlgebra
	using StatsBase, StatsPlots
	using Turing, ForwardDiff, Distributions
	using SymbolicRegression
end

# ╔═╡ a3695b0f-0114-4bfd-95df-976b7ce7917b
md"""
# Workings
"""

# ╔═╡ d8869f3e-dd69-4c54-a50c-a92a18dcd553
begin
	Turing.setadbackend(:forwarddiff)
	rng = MersenneTwister(100)
	NN_seed = Flux.glorot_uniform(rng)
end

# ╔═╡ 203015cd-2f2e-489b-98a5-3303ed8ff740
TableOfContents()

# ╔═╡ 70363142-abaa-4173-8c43-c7cca2cbfd65
md"""
## Loading Data
"""

# ╔═╡ 9ae356d7-a73f-473d-897a-a20773ddb1a1
md"""
Loading data from HMD for Ireland.
"""

# ╔═╡ ab894bf3-3bc4-4942-a792-bdd9b2ce51ee
begin 
	df = DataFrame(CSV.File("../data/irish_mx.csv"))
	
	# Reset the index such that t=0 is 1950
	df.t = df.t .- 1950

	# Exclude very old ages (100+)
	df = df[df.x .≤ 100, :]
end

# ╔═╡ 8200a4c5-2e52-460f-a119-55c5df12018b
begin
	p_1950 = begin 
		plot(xlab="Age", ylab="log(mₓ)", title="log(mₓ) Rates for Ireland: \n1950")
		scatter!(df[df.t .== 0, :x], log.(df[df.t .== 0, :mx_m]), label="Males")
		scatter!(df[df.t .== 0, :x], log.(df[df.t .== 0, :mx_f]), label="Females")
	end

	p_1960 = begin 
		plot(xlab="Age", ylab="log(mₓ)", title="log(mₓ) Rates for Ireland: \n1960")
		scatter!(df[df.t .== 10, :x], log.(df[df.t .== 10, :mx_m]), label="Males")
		scatter!(df[df.t .== 10, :x], log.(df[df.t .== 10, :mx_f]), label="Females")
	end

	plot(p_1950, p_1960)
end

# ╔═╡ edbd39a6-40ef-4b8b-ab1b-b212af24f8ff
begin
	p_2000 = begin 
		plot(xlab="Age", ylab="log(mₓ)", title="log(mₓ) Rates for Ireland: \n2000")
		scatter!(df[df.t .== 50, :x], log.(df[df.t .== 50, :mx_m]), label="Males")
		scatter!(df[df.t .== 50, :x], log.(df[df.t .== 50, :mx_f]), label="Females")
	end

	p_2020 = begin 
		plot(xlab="Age", ylab="log(mₓ)", title="log(mₓ) Rates for Ireland: \n2020")
		scatter!(df[df.t .== 70, :x], log.(df[df.t .== 70, :mx_m]), label="Males")
		scatter!(df[df.t .== 70, :x], log.(df[df.t .== 70, :mx_f]), label="Females")
	end

	plot(p_2000, p_2020)
end

# ╔═╡ eaeee271-f399-42b5-ae18-11c5b5d12f7b
md"""
Want to find an approximation for $f(x, t, g)$ which can produces a mortality rate.
"""

# ╔═╡ 25fb34e6-8043-4c4e-ad9f-729be34d472d
begin
	plot(xlab="Age (x+20)", ylab="Period (t+1950)", title="Heatmap:\n1950-2020")
	heatmap!(Matrix{Float64}(select(unstack(df[df.x .> 19 .&& df.x .< 56, :], :t, :x, :mx_f), Not(:t))))
end

# ╔═╡ 2e3e0c9f-1aa6-4336-b7c6-e0a52b783e06
begin
	# Create an indicator for Gender
	df.g .= 0
	
	# Split out males
	df_m = df[:, [:t, :x, :g, :mx_m]]
	df_m.g .= 1
	rename!(df_m, :mx_m => "mx")
	
	# Split out females
	df_f = df[:, [:t, :x, :g, :mx_f]]
	df_f.g .= 0
	rename!(df_f, :mx_f => "mx")

	# Recombine
	df_comb = outerjoin(df_m, df_f, on=[:t, :x, :mx, :g])

	# Filter out zero mortality
	filter!(:mx => (d) -> d > 0, df_comb)
end

# ╔═╡ 00a2e31e-11db-448a-accc-b578ea77f0c2
train, test = splitobs(
	shuffle(
		rng,
		df_comb[:, [:t, :x, :g, :mx]]
	);
	at = 0.7
);

# ╔═╡ b3e4caed-0a75-4261-b0ec-eceec1ba3f87
begin
	X_train = Matrix{Float32}(train[:, Not(:mx)])
	y_train = Vector{Float32}(train.mx)
	X_test = Matrix{Float32}(test[:, Not(:mx)])
	y_test = Vector{Float32}(test.mx)
end

# ╔═╡ b79c5d4d-c38d-4db9-aea3-bdd4a7287d77
X_train

# ╔═╡ 66b76719-9638-472f-b078-5aee93a0ebef
md"""
## Case #1: NN with Variance Prior
"""

# ╔═╡ 623974eb-3555-491f-a07b-42aa588bf5bc
MLP_model = Chain(
	Dense(3 => 32, tanh; init=NN_seed),
	Dense(32 => 64, tanh; init=NN_seed),
	Dense(64 => 16, tanh; init=NN_seed),
	Dense(16 => 1, σ; init=NN_seed)
)

# ╔═╡ 8953cc5d-48a7-496c-82f6-fbd56ac0c727
begin
	opt_params = Flux.setup(Flux.Adam(), MLP_model)
	epochs = 15_000
	loss(x, y) = Flux.Losses.mse(x, y)
	losses_train = zero(rand(epochs))
	losses_test = zero(rand(epochs))
end;

# ╔═╡ 325c59a7-f1a5-4cc8-a52a-de6786e2e094
# ╠═╡ show_logs = false
for epoch ∈ 1:epochs
	Flux.train!(MLP_model, [(X_train', y_train')], opt_params) do m, x, y
    	loss(m(x), y)
  	end
	losses_train[epoch] = loss(MLP_model(X_train'), y_train')
	losses_test[epoch] = loss(MLP_model(X_test'), y_test')
end

# ╔═╡ 8b90495e-6c40-4cd3-bab1-df52a0397ab9
(losses_train[epochs], losses_test[epochs])

# ╔═╡ 349264ab-7c30-4310-b785-2bbbf8c18b97
begin
	plot(title="Loss (MSE)\n Starting at 1 000 Epochs", xlabel="Epochs", ylabel="Loss (MSE)", legend=true)
	scatter!(1_000:epochs, losses_train[1_000:epochs], color="purple", markeralpha=0.25, label="Training")
	scatter!(1_000:epochs, losses_test[1_000:epochs], color="orange", markeralpha=0.25, label="Testing")
end

# ╔═╡ cd5f68fa-5f85-40b7-b328-6e7242cb31be
test.NN_pred = (MLP_model(X_test')')[:]

# ╔═╡ d03c8e8f-b730-4587-ba64-5c26a8fda32c
t1 = combine(
	groupby(
		test, [:x]
		),
	[:mx, :NN_pred] .=> mean; renamecols=false)

# ╔═╡ b92cdc1c-3831-4c08-968c-4b92a9543f85
begin
	plot(xlab="Age", ylab="log(mₓ)")
	scatter!(t1.x, log.(t1.mx), label="Testing")
	plot!(t1.x, log.(t1.NN_pred), label="NN", width=2)
end

# ╔═╡ 77d920ba-e76f-45d3-a0f7-15bbe9772413
@model function BNN_1(X::Matrix{Float32}, y::Vector{Float32}, nn::Chain, ::Type{T} = Float32) where {T}
	# priors
	α ~ truncated(Normal(0, 5); lower=0)
	β ~ truncated(Normal(0, 5); lower=0)
	σ ~ InverseGamma(α, β)

	μ = (nn(X')')[:]

	# Likelihood
	y ~ MvNormal(μ, σ * I)
	return Nothing
end

# ╔═╡ 6ca6b971-798b-471e-8428-b72aad87bfcb
begin
	num_chains_1 = 4
	sample_size_1 = 5_000
end;

# ╔═╡ 4907f750-d1d4-4846-b359-7fee0338f734
NN_1_samples = sample(
	BNN_1(
		X_train, 
		y_train, 
		MLP_model
	), 
	NUTS(),
	MCMCThreads(),
	sample_size_1,
	num_chains_1;
	discard_adapt = false
);

# ╔═╡ 0b2015dd-bd0c-4a84-8951-ffd80fa836f1
describe(NN_1_samples)

# ╔═╡ 575e14db-717c-408e-8b91-007da4a0a2e4
plot(NN_1_samples)

# ╔═╡ b01a799c-c407-4540-9bb2-3fc0cefd4ccc
plot(NN_1_samples[2_500:end, :, :])

# ╔═╡ 455674fc-188e-4c2a-a743-e4a52ed77e55
begin
	plot(xlab="Age", ylab="mₓ", xlim=(20, 65), ylim=(-.001, .025))
	for i ∈ 1_000:sample_size_1
		σ_1_::Float32 = mode(MCMCChains.group(NN_1_samples[i:i, :, :], :σ).value)
		plot!(t1.x, t1.NN_pred, label=false, width=.1, ribbon=σ_1_, colour="orange", fillalpha=.005)
	end
	scatter!(t1.x, t1.mx, label="Testing", colour="blue")
	plot!(t1.x, t1.NN_pred, label="NN", width=2, colour="red")
end

# ╔═╡ Cell order:
# ╟─a3695b0f-0114-4bfd-95df-976b7ce7917b
# ╠═0b0bb0d0-7d73-11ee-21ff-79bf038dbc5b
# ╠═ee32c415-f6c5-49cd-97de-0213f01d0ec6
# ╠═d8869f3e-dd69-4c54-a50c-a92a18dcd553
# ╟─203015cd-2f2e-489b-98a5-3303ed8ff740
# ╟─70363142-abaa-4173-8c43-c7cca2cbfd65
# ╟─9ae356d7-a73f-473d-897a-a20773ddb1a1
# ╠═ab894bf3-3bc4-4942-a792-bdd9b2ce51ee
# ╠═8200a4c5-2e52-460f-a119-55c5df12018b
# ╠═edbd39a6-40ef-4b8b-ab1b-b212af24f8ff
# ╟─eaeee271-f399-42b5-ae18-11c5b5d12f7b
# ╠═25fb34e6-8043-4c4e-ad9f-729be34d472d
# ╠═2e3e0c9f-1aa6-4336-b7c6-e0a52b783e06
# ╠═00a2e31e-11db-448a-accc-b578ea77f0c2
# ╠═b3e4caed-0a75-4261-b0ec-eceec1ba3f87
# ╠═b79c5d4d-c38d-4db9-aea3-bdd4a7287d77
# ╟─66b76719-9638-472f-b078-5aee93a0ebef
# ╠═623974eb-3555-491f-a07b-42aa588bf5bc
# ╠═8953cc5d-48a7-496c-82f6-fbd56ac0c727
# ╠═325c59a7-f1a5-4cc8-a52a-de6786e2e094
# ╠═8b90495e-6c40-4cd3-bab1-df52a0397ab9
# ╠═349264ab-7c30-4310-b785-2bbbf8c18b97
# ╠═cd5f68fa-5f85-40b7-b328-6e7242cb31be
# ╠═d03c8e8f-b730-4587-ba64-5c26a8fda32c
# ╠═b92cdc1c-3831-4c08-968c-4b92a9543f85
# ╠═77d920ba-e76f-45d3-a0f7-15bbe9772413
# ╠═6ca6b971-798b-471e-8428-b72aad87bfcb
# ╠═4907f750-d1d4-4846-b359-7fee0338f734
# ╠═0b2015dd-bd0c-4a84-8951-ffd80fa836f1
# ╠═575e14db-717c-408e-8b91-007da4a0a2e4
# ╠═b01a799c-c407-4540-9bb2-3fc0cefd4ccc
# ╠═455674fc-188e-4c2a-a743-e4a52ed77e55
