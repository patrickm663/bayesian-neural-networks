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
	using Flux, Random, MLDataUtils, CUDA
	using CSV, DataFrames, LinearAlgebra
	using StatsBase, StatsPlots
	using Turing, ForwardDiff, Distributions
	using SymbolicRegression, Latexify
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

# ╔═╡ ed87520b-328c-4c0f-9bf3-e500bce7c2f4
md"""
Choose $t=65 \equiv 2015$ as a cut-off between training and testing.
"""

# ╔═╡ 00a2e31e-11db-448a-accc-b578ea77f0c2
begin
	train = df_comb[20 .< df_comb.t .≤ 60, :]
	test = df_comb[df_comb.t .> 60, :]
end;

# ╔═╡ b3e4caed-0a75-4261-b0ec-eceec1ba3f87
begin
	X_train_ = Matrix{Float32}(train[:, Not(:mx)])
	y_train = log.(Vector{Float32}(train.mx))
	X_test_ = Matrix{Float32}(test[:, Not(:mx)])
	y_test = log.(Vector{Float32}(test.mx))

	function z_scale(data; μₜ=mean(X_train_[:, 1]), σₜ=std(X_train_[:, 1]), μₓ=mean(X_train_[:, 2]), σₓ=std(X_train_[:, 2]))
		data_ = deepcopy(data)
		data_[:, 1] = (data[:, 1] .- μₜ) ./ σₜ
		data_[:, 2] = (data[:, 2] .- μₓ) ./ σₓ
		return data_
	end

	X_train = z_scale(X_train_)
	X_test = z_scale(X_test_)
end

# ╔═╡ 278a7d2d-9c36-49d5-92e7-4a49c6848812
z_scale(X_test_)

# ╔═╡ 842b148c-2088-4032-87c3-a1b85af936d6
X_test_

# ╔═╡ 66b76719-9638-472f-b078-5aee93a0ebef
md"""
## Case #1: Vanilla NN with Constant Variance
"""

# ╔═╡ 2e6d8260-0e48-42f7-a3fb-69d442dcff03
md"""
This is a fully data-driven approach whereby the model learns all that it needs to from the data.
"""

# ╔═╡ 623974eb-3555-491f-a07b-42aa588bf5bc
MLP_model = Chain(
	Dense(3 => 128, tanh; init=NN_seed),
	Dense(128 => 128, tanh; init=NN_seed),
	Dense(128 => 128, tanh; init=NN_seed),
	Dense(128 => 128, tanh; init=NN_seed),
	Dense(128 => 1, identity; init=NN_seed)
) |> gpu

# ╔═╡ 8953cc5d-48a7-496c-82f6-fbd56ac0c727
begin
	opt_params = Flux.setup(Flux.Adam(), MLP_model)
	epochs = 30_000
	loss(x, y) = Flux.Losses.mse(x, y)
	losses_train = zero(rand(epochs))
	losses_test = zero(rand(epochs))
end;

# ╔═╡ 325c59a7-f1a5-4cc8-a52a-de6786e2e094
# ╠═╡ show_logs = false
for epoch ∈ 1:epochs
	Flux.train!(MLP_model, [(X_train' |> gpu, y_train' |> gpu)], opt_params) do m, x, y
    	loss(m(x), y)
  	end
	losses_train[epoch] = loss(MLP_model(X_train' |> gpu) |> cpu, y_train')
	losses_test[epoch] = loss(MLP_model(X_test' |> gpu) |> cpu, y_test')
end

# ╔═╡ 8b90495e-6c40-4cd3-bab1-df52a0397ab9
(losses_train[epochs], losses_test[epochs])

# ╔═╡ 349264ab-7c30-4310-b785-2bbbf8c18b97
begin
	plot(title="Loss (MSE)\n Starting at 5 000 Epochs", xlabel="Epochs", ylabel="Loss (MSE)", legend=true)
	scatter!(5_000:epochs, losses_train[5_000:end], color="purple", markeralpha=0.25, label="Training")
	scatter!(5_000:epochs, losses_test[5_000:end], color="orange", markeralpha=0.25, label="Testing")
end

# ╔═╡ cd5f68fa-5f85-40b7-b328-6e7242cb31be
test.NN_pred = (MLP_model(X_test' |> gpu)' |> cpu)[:]

# ╔═╡ cccd9c01-78f5-491a-b4cc-fd87da919c74
test

# ╔═╡ d03c8e8f-b730-4587-ba64-5c26a8fda32c
t1 = combine(
	groupby(
		test[:, :], [:x]
		),
	[:mx, :NN_pred] .=> mean; renamecols=false)

# ╔═╡ b92cdc1c-3831-4c08-968c-4b92a9543f85
begin
	plot(xlab="Age", ylab="log(mₓ)")
	scatter!(t1.x, log.(t1.mx), label="Testing")
	plot!(t1.x, (t1.NN_pred), label="NN", width=2)
end

# ╔═╡ b16da85a-8e1a-4c9a-ad55-15b5f014ae42
md"""
Currently, there is a slight overfit in younger ages. A deeper network with more training time may resolve this.
"""

# ╔═╡ 77d920ba-e76f-45d3-a0f7-15bbe9772413
@model function BNN_1(X::Matrix{Float32}, y::Vector{Float32}, nn::Chain, ::Type{T} = Float32) where {T}
	# priors
	α ~ truncated(Normal(0, 5); lower=0.0001)
	β ~ truncated(Normal(0, 5); lower=0.0001)
	σ ~ InverseGamma(α, β)

	μ = (nn(X')')[:]

	# Likelihood
	y ~ MvNormal(μ, σ * I)
	return Nothing
end

# ╔═╡ 6ca6b971-798b-471e-8428-b72aad87bfcb
begin
	num_chains_1 = 2
	sample_size_1 = 5_000
end;

# ╔═╡ 4907f750-d1d4-4846-b359-7fee0338f734
NN_1_samples = sample(
	BNN_1(
		X_train, 
		y_train, 
		MLP_model |> cpu
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
	plot(xlab="Age", ylab="log(mₓ)")
	for i ∈ 10:sample_size_1
		σ_1_::Float32 = mode(MCMCChains.group(NN_1_samples[i:i, :, :], :σ).value)
		plot!(t1.x, t1.NN_pred, label=false, width=.1, ribbon=σ_1_, colour="orange", fillalpha=.5)
	end
	scatter!(t1.x, log.(t1.mx), label="Testing", colour="blue")
	plot!(t1.x, t1.NN_pred, label="NN", width=2, colour="red")
end

# ╔═╡ d597c87b-54b6-4823-83fd-7d73b6d8b0a2
md"""
We now want to derive an analytical expression for this model wrt age, time period, and gender.
"""

# ╔═╡ 31a3d181-eb05-4e55-941b-34f826df69eb
options = SymbolicRegression.Options(
    binary_operators=[+, *, -, /, ^],
    unary_operators=[exp, log, abs],
    npop=75,
	should_simplify=true,
	seed=1001
);

# ╔═╡ 125e7009-1d85-4426-83a2-dec18e6189b7
# ╠═╡ show_logs = false
hall_of_fame = EquationSearch(
    X_train', (MLP_model(X_train' |> gpu)' |> cpu)[:], niterations=50,
	options=options,
    parallelism=:multithreading
);

# ╔═╡ 09657e94-a0c7-4432-981c-4e3c27e2d6a4
dominating = calculate_pareto_frontier(hall_of_fame)

# ╔═╡ 11920c8c-af7d-4d9e-a0bb-785b18ed2ed6
trees = [member.tree for member in dominating]

# ╔═╡ dec3e545-0529-42ec-8e11-4c9f8d2666f8
begin
	test.Sym_Pred_simple, _ = eval_tree_array(trees[9], X_test', options)
	test.Sym_Pred_complex, _ = eval_tree_array(trees[end], X_test', options)
end

# ╔═╡ d78733bf-6aa1-49dd-8849-6755cd55005e
mean(X_train_[:, 1])

# ╔═╡ 175b49c9-aede-4848-ad83-3567860c8d06
mean(X_train_[:, 2])

# ╔═╡ 71683735-8a1f-4d68-92f5-2f65186f5c34
latexify(
	replace(
		"log(m(x,t,g))= " * string(trees[9]), 
		"x1" => "((t-μ_t)/σ_t)",
		"x2" => "((x-μ_x)/σ_x)",
		"x3" => "g"
	)
)

# ╔═╡ 3ec84ccf-eec6-4860-adf8-205181b6ee34
latexify(
	replace(
		"log(m(x,t,g))= " * string(trees[end]), 
		"x1" => "((t-μ_t)/σ_t)",
		"x2" => "((x-μ_x)/σ_x)",
		"x3" => "g"
	)
)

# ╔═╡ b831c9bf-7965-4ad5-b247-ed97d8e066d7
test

# ╔═╡ d99f1ffd-e417-4f78-be5d-d193ead56932
t1_ = combine(
	groupby(
		test, [:x]
		),
	[:mx, :NN_pred, :Sym_Pred_simple, :Sym_Pred_complex] .=> mean; renamecols=false)

# ╔═╡ 8309a20c-294c-4024-83a8-432cacaa032a
begin
	plot(xlab="Age", ylab="log(mₓ)")
	for i ∈ 1:sample_size_1
		σ_1_::Float32 = mode(MCMCChains.group(NN_1_samples[i:i, :, :], :σ).value)
		plot!(t1_.x, t1_.NN_pred, label=false, width=.1, ribbon=σ_1_, colour="orange", fillalpha=.005)
	end
	scatter!(t1_.x, log.(t1_.mx), label="Testing", colour="blue")
	plot!(t1_.x, t1_.NN_pred, label="NN", width=2, colour="red")
	plot!(t1_.x, t1_.Sym_Pred_simple, label="Sym Simple", width=2, colour="green")
	plot!(t1_.x, t1_.Sym_Pred_complex, label="Sym Complex", width=2, colour="black")
end

# ╔═╡ 759e685c-3967-4672-ac24-f15f632c07e0
# forecast some data for future years

# ╔═╡ f41c15b8-1f73-4712-9709-2bc1e39a096f
md"""
## Case #2: Full BNN
"""

# ╔═╡ 9cc4db12-8bf0-4a36-90df-cb34e498ae2a
md"""
A smaller network is used due to run-time constraints (a smaller sample of data should be compared)
"""

# ╔═╡ c4a66d74-7c5a-4dfa-90b2-46c8340e2393
BNN_model = Chain(
	Dense(3 => 4, tanh; init=NN_seed),
	Dense(4 => 2, tanh; init=NN_seed),
	Dense(2 => 1, identity; init=NN_seed)
)

# ╔═╡ 72908874-37de-4ace-a429-49eb79b75f5b
@model function BNN_2(X::Matrix{Float32}, y::Vector{Float32}, nn::Chain, ::Type{T} = Float32) where {T}
	param_initial, reconstruct = Flux.destructure(nn)
	n_parameters = length(param_initial)

	# Priors
	parameters ~ MvNormal(zeros(n_parameters), 1)
	σ ~ truncated(Normal(0, 5); lower=0.0001)

	# Reconstruct the neural network and retrieve a set of observations
    nn = reconstruct(parameters)
    μ = nn(X')

	# Create a prior on our log mortality using a constant variance and neural network output (based on priors on the weights) 
	y ~ MvNormal(vec(μ), σ * I)
	return Nothing
end

# ╔═╡ 42659c1b-ca9d-4eb4-9d28-f9779e4195ea
begin
	num_chains_2 = 2
	sample_size_2 = 2_000
end;

# ╔═╡ eb9a6c3a-2e0e-44ba-9ff7-e3f2b964e6ca
NN_2_samples = sample(
	BNN_2(
		X_train, 
		y_train, 
		BNN_model
	), 
	NUTS(),
	MCMCThreads(),
	sample_size_2,
	num_chains_2;
	discard_adapt = false
);

# ╔═╡ a6ef9635-de4b-40a7-87e3-f710b3f9ea76
describe(NN_2_samples)

# ╔═╡ 6636da6b-4206-46a6-89a6-3b71bf2b958f
plot(NN_2_samples[1_000:end, 2:10:end, :])

# ╔═╡ 650bffcd-c3df-4006-8119-b192338b3a43
md"""
No clear convergence. Between chains, sometimes shrink weights to zero. More samples likely needed.
"""

# ╔═╡ 3570bff7-2e69-4426-8ca0-48b1feb1c015
θ_samples = MCMCChains.group(NN_2_samples, :parameters).value

# ╔═╡ d68d6157-c0cc-4a53-8ce4-b6d041831a4d
vec(θ_samples.data[end:end, :, :][:, :, 1])

# ╔═╡ 06f0b019-dda9-4f2b-9312-f46040f36e3c
md"""
## Case #3: Variational Infernence
"""

# ╔═╡ 19594454-6031-407e-9944-8a25b80de8fa
BNN_model_ = Chain(
	Dense(3 => 4, tanh; init=NN_seed),
	Dense(4 => 8, tanh; init=NN_seed),
	Dense(8 => 8, tanh; init=NN_seed),
	Dense(8 => 1, identity; init=NN_seed)
)

# ╔═╡ 1dfc7bb3-90e2-40ff-8850-e7e82c8b05dd
m_ = BNN_2(
		X_train, 
		y_train, 
		BNN_model_
	)

# ╔═╡ 711925fb-d125-477b-ad4e-a35ddd3b12dc
# ╠═╡ show_logs = false
q__ = vi(m_, ADVI(10, 1_000))

# ╔═╡ 55b65ab1-46d1-4e12-9f2f-f93bb1e23e64
z__ = rand(q__, 20_000);

# ╔═╡ 68603969-c803-485f-af4e-e53dde1e7eb4
z__

# ╔═╡ 9232fe02-2341-45bd-8d5f-58a50d4eaa4a
function plot_variational_marginals(z, sym2range)
    ps = []

    for (i, sym) ∈ enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            offset = 1
            for r ∈ indices
                p = density(
                    z[r, :];
                    title="$(sym)[$offset]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density"
                )
                push!(ps, p)
                offset += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density"
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 10000))
end

# ╔═╡ 35f85c5f-96f6-4196-9b8d-f6d26dc12bfa
_, sym2range = bijector(m_, Val(true));

# ╔═╡ eedb0b3a-a3fa-4617-8a4f-6b97d0f67c9e
#plot_variational_marginals(z__, sym2range)

# ╔═╡ c1d8080b-a68b-4fa4-93ea-5b976a1f2739
nn_forward(x, θ, nn) = Flux.destructure(nn)[2](θ)(x')

# ╔═╡ 28dad57c-9da3-459d-8273-4530adf18498
nn_forward(X_test, vec(θ_samples.data[end:end, :, :][:, :, 1]), BNN_model)'[:]

# ╔═╡ 1d765694-4192-4244-9267-fd352d1823d1
begin
	plot(xlab="Age", ylab="log(mₓ)", title="BNN (Green) across Multiples Samples", label="BNN")
	for i ∈ 100:sample_size_2
		for j ∈ 1:num_chains_2
			test.BNN_temp = nn_forward(
				X_test, 
				vec(θ_samples.data[i:i, :, :][:, :, j]), 
				BNN_model)'[:]
			t1____ = combine(
				groupby(
					test, [:x]
					),
				[:mx, :BNN_temp] .=> mean; renamecols=false)
			σ_2_ = mode(MCMCChains.group(NN_2_samples[i:i, :, j], :σ).value)
			plot!(t1____.x, t1____.BNN_temp, label=false, width=.1, colour="green", alpha=.5, ribbon=σ_2_, fillalpha=.05)
		end
	end
	scatter!(t1_.x, log.(t1_.mx), label="Testing", colour="blue")
	plot!(t1_.x, t1_.NN_pred, label="NN", width=2, colour="red")
end

# ╔═╡ 78550e95-fe66-49b2-81aa-925b7c504f82
begin
	plot(xlab="Age", ylab="log(mₓ)", title="Variational Inference over an NN (Orange)")
	for i ∈ 8_000:size(z__)[2]
		test.VI_temp = nn_forward(
						X_test,
						vec(z__[1:(end-1), i]),
						BNN_model_)'[:]
		t1___ = combine(
			groupby(
				test, [:x]
				),
			[:mx, :VI_temp] .=> mean; renamecols=false)
		plot!(t1___.x, t1___.VI_temp, label=false, width=.1, colour="orange", alpha=.5)
	end
	scatter!(t1_.x, log.(t1_.mx), label="Testing", colour="blue")
	plot!(t1_.x, t1_.NN_pred, label="NN", width=2, colour="red")
end

# ╔═╡ 66e97f44-5854-4194-85b6-f12846ab41b1
md"""
## Case #4: BNN into Existing Mortality Model Structure
"""

# ╔═╡ cc81628d-e28f-4c60-a49e-ef7892506498


# ╔═╡ 11fee4c8-2a1b-4c97-95d5-d43bc8877d64


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
# ╟─ed87520b-328c-4c0f-9bf3-e500bce7c2f4
# ╠═00a2e31e-11db-448a-accc-b578ea77f0c2
# ╠═b3e4caed-0a75-4261-b0ec-eceec1ba3f87
# ╠═278a7d2d-9c36-49d5-92e7-4a49c6848812
# ╠═842b148c-2088-4032-87c3-a1b85af936d6
# ╟─66b76719-9638-472f-b078-5aee93a0ebef
# ╟─2e6d8260-0e48-42f7-a3fb-69d442dcff03
# ╠═623974eb-3555-491f-a07b-42aa588bf5bc
# ╠═8953cc5d-48a7-496c-82f6-fbd56ac0c727
# ╠═325c59a7-f1a5-4cc8-a52a-de6786e2e094
# ╠═8b90495e-6c40-4cd3-bab1-df52a0397ab9
# ╠═349264ab-7c30-4310-b785-2bbbf8c18b97
# ╠═cd5f68fa-5f85-40b7-b328-6e7242cb31be
# ╠═cccd9c01-78f5-491a-b4cc-fd87da919c74
# ╠═d03c8e8f-b730-4587-ba64-5c26a8fda32c
# ╠═b92cdc1c-3831-4c08-968c-4b92a9543f85
# ╟─b16da85a-8e1a-4c9a-ad55-15b5f014ae42
# ╠═77d920ba-e76f-45d3-a0f7-15bbe9772413
# ╠═6ca6b971-798b-471e-8428-b72aad87bfcb
# ╠═4907f750-d1d4-4846-b359-7fee0338f734
# ╠═0b2015dd-bd0c-4a84-8951-ffd80fa836f1
# ╠═575e14db-717c-408e-8b91-007da4a0a2e4
# ╠═b01a799c-c407-4540-9bb2-3fc0cefd4ccc
# ╠═455674fc-188e-4c2a-a743-e4a52ed77e55
# ╟─d597c87b-54b6-4823-83fd-7d73b6d8b0a2
# ╠═31a3d181-eb05-4e55-941b-34f826df69eb
# ╠═125e7009-1d85-4426-83a2-dec18e6189b7
# ╠═09657e94-a0c7-4432-981c-4e3c27e2d6a4
# ╠═11920c8c-af7d-4d9e-a0bb-785b18ed2ed6
# ╠═dec3e545-0529-42ec-8e11-4c9f8d2666f8
# ╠═d78733bf-6aa1-49dd-8849-6755cd55005e
# ╠═175b49c9-aede-4848-ad83-3567860c8d06
# ╠═71683735-8a1f-4d68-92f5-2f65186f5c34
# ╠═3ec84ccf-eec6-4860-adf8-205181b6ee34
# ╠═b831c9bf-7965-4ad5-b247-ed97d8e066d7
# ╠═d99f1ffd-e417-4f78-be5d-d193ead56932
# ╠═8309a20c-294c-4024-83a8-432cacaa032a
# ╠═759e685c-3967-4672-ac24-f15f632c07e0
# ╟─f41c15b8-1f73-4712-9709-2bc1e39a096f
# ╟─9cc4db12-8bf0-4a36-90df-cb34e498ae2a
# ╠═c4a66d74-7c5a-4dfa-90b2-46c8340e2393
# ╠═72908874-37de-4ace-a429-49eb79b75f5b
# ╠═42659c1b-ca9d-4eb4-9d28-f9779e4195ea
# ╠═eb9a6c3a-2e0e-44ba-9ff7-e3f2b964e6ca
# ╠═a6ef9635-de4b-40a7-87e3-f710b3f9ea76
# ╠═6636da6b-4206-46a6-89a6-3b71bf2b958f
# ╟─650bffcd-c3df-4006-8119-b192338b3a43
# ╠═3570bff7-2e69-4426-8ca0-48b1feb1c015
# ╠═d68d6157-c0cc-4a53-8ce4-b6d041831a4d
# ╠═28dad57c-9da3-459d-8273-4530adf18498
# ╠═1d765694-4192-4244-9267-fd352d1823d1
# ╟─06f0b019-dda9-4f2b-9312-f46040f36e3c
# ╠═19594454-6031-407e-9944-8a25b80de8fa
# ╠═1dfc7bb3-90e2-40ff-8850-e7e82c8b05dd
# ╠═711925fb-d125-477b-ad4e-a35ddd3b12dc
# ╠═55b65ab1-46d1-4e12-9f2f-f93bb1e23e64
# ╠═68603969-c803-485f-af4e-e53dde1e7eb4
# ╠═9232fe02-2341-45bd-8d5f-58a50d4eaa4a
# ╠═35f85c5f-96f6-4196-9b8d-f6d26dc12bfa
# ╠═eedb0b3a-a3fa-4617-8a4f-6b97d0f67c9e
# ╠═c1d8080b-a68b-4fa4-93ea-5b976a1f2739
# ╠═78550e95-fe66-49b2-81aa-925b7c504f82
# ╟─66e97f44-5854-4194-85b6-f12846ab41b1
# ╠═cc81628d-e28f-4c60-a49e-ef7892506498
# ╠═11fee4c8-2a1b-4c97-95d5-d43bc8877d64
