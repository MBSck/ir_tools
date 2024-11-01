import PhysicalConstants.CODATA2018: c_0, h, k_B

using Interpolations
using NPZ
using ProgressMeter
using Roots
using Trapz
using Unitful
using UnitfulAngles
using UnitfulAstro

function BlackBody(ν::Float64, T::Float64)::Float64
  ustrip(uconvert(u"erg/s/cm^2/Hz/sr", 2h * (ν * u"Hz")^3 / (exp(h * (ν * u"Hz") / (k_B * T * u"K")) - 1) / c_0^2))
end

p_out(κ::Vector{Float64}, ν::Vector{Float64}, T::Float64)::Float64 = mapslices(x -> trapz(ν, x), BlackBody.(ν, T) .* κ, dims=1)[begin]
thermal_equilibrium(p_in::Float64, κ::Vector{Float64}, ν::Vector{Float64}, T::Float64)::Float64 = p_in - p_out(κ, ν, T)

function compute_temperature_grid(
  wavelengths::Vector{Float64},
  distance::Float64,
  flux_star::Vector{Float64},
  silicate_opacity::Vector{Float64},
  continuum_opacity::Vector{Float64},
  radial_range::Vector{Float64}=[0.1, 100.0],
  radial_dim::Int=2048,
  temperature_range::Vector{Int}=[0, 10000],
  weight_steps::Float64=0.01,
  ncores::Int=6,
)
  ν = @. ustrip(c_0 / (wavelengths * u"m"))
  flux_star = @. uconvert(u"erg/s/cm^2/Hz", flux_star * u"Jy")
  radii = 10 .^ range(log10(radial_range[begin]), log10(radial_range[end]), length=radial_dim)
  radiation = @. ustrip(flux_star * (uconvert(u"AU", distance * u"pc") / (radii * u"AU")')^2)

  weights = 0:weight_steps:1.0
  κ_abs = [@. (1 - w) * silicate_opacity + w * continuum_opacity for w in weights]

  p_in = Array{Float64}(undef, length(κ_abs), length(radii))
  for (i, κ) in enumerate(κ_abs)
    p_in[i, :] = mapslices(x -> trapz(ν, x), radiation .* κ, dims=1)'
  end

  progress = Progress(length(weights), desc="Computing weight-temperature matrix...")
  weight_grid = Array{Float64}(undef, (size(weights)..., size(radii)...))

  Threads.@threads for i in eachindex(weights)
    for j in eachindex(radii)
      weight_grid[i, j] = find_zero(T -> thermal_equilibrium(p_in[i, j], κ_abs[i], ν, T), temperature_range)
    end
    next!(progress)
  end
  npzwrite("weight_temperatures.npy", weight_grid)
  npzwrite("radii.npy", radii)
  finish!(progress)
end


if abspath(PROGRAM_FILE) == @__FILE__
  data_dir = joinpath(homedir(), "data")
  opacity_dir = joinpath(data_dir, "opacities")

  method = "grf"
  wl_op, silicate_op = eachrow(npzread(joinpath(opacity_dir, "hd142527_silicate_$(method)_opacities.npy")))
  wl_flux, flux = eachrow(npzread(joinpath(data_dir, "flux", "hd142527", "HD142527_stellar_model.npy")))
  wl_cont, cont_op = eachrow(npzread(joinpath(data_dir, "opacities", "qval", "Q_amorph_c_rv0.1.npy")))

  # NOTE: Constrain the flux wavlengths
  indices = max(wl_op[begin], wl_cont[begin]) .< wl_flux .< min(wl_op[end], wl_cont[end])
  wl_flux, flux = wl_flux[indices], flux[indices]

  silicate_op = linear_interpolation(wl_op, silicate_op, extrapolation_bc=Line())(wl_flux)
  cont_op = linear_interpolation(wl_cont, cont_op, extrapolation_bc=Line())(wl_flux)

  compute_temperature_grid(wl_flux * 1e-6, 158.51, flux, silicate_op, cont_op)
end
