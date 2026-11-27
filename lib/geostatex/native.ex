defmodule Geostatex.Native do
  @moduledoc false
  # NIF bindings to the `geostatex_native` Rust crate. Each function below is replaced when the
  # NIF library loads; the bodies only run if loading failed.
  #
  # The library is built from source with Rustler unless a release ships precompiled artifacts:
  # building is forced when GEOSTATEX_BUILD is "1"/"true", when the version is a pre-release, or
  # when no checksum file ships with the package.
  version = Mix.Project.config()[:version]

  force_build? =
    System.get_env("GEOSTATEX_BUILD") in ["1", "true"] or
      Version.parse!(version).pre != [] or
      not File.exists?("checksum-Elixir.Geostatex.Native.exs")

  use RustlerPrecompiled,
    otp_app: :geostatex,
    crate: "geostatex_native",
    version: version,
    base_url: "https://github.com/CountlinkX-Solutions/geostatex/releases/download/v#{version}",
    force_build: force_build?,
    nif_versions: ["2.16", "2.17"],
    targets: ~w(
      aarch64-apple-darwin
      aarch64-unknown-linux-gnu
      x86_64-apple-darwin
      x86_64-pc-windows-msvc
      x86_64-unknown-linux-gnu
      x86_64-unknown-linux-musl
    )

  def nif_version, do: :erlang.nif_error(:nif_not_loaded)
  def sum_f64(_binary), do: :erlang.nif_error(:nif_not_loaded)

  # Variograms. Numeric vectors are f64 binaries.
  def variogram_experimental(_coords, _dim, _values, _dist, _maxlag, _bins, _est, _pairs),
    do: :erlang.nif_error(:nif_not_loaded)

  def spacetime_experimental(_coords, _values, _dist, _x_spec, _maxlag, _t_spec, _estimator),
    do: :erlang.nif_error(:nif_not_loaded)

  def variogram_fit(_bins, _experimental, _model, _method, _sigma, _use_nugget),
    do: :erlang.nif_error(:nif_not_loaded)

  def variogram_model_values(_model, _cof, _h), do: :erlang.nif_error(:nif_not_loaded)

  # Ordinary kriging (resource).
  def kriging_new(_coords, _values, _model, _params, _fit, _solve, _anis, _drifts),
    do: :erlang.nif_error(:nif_not_loaded)

  def kriging_execute(_ref, _targets, _extra, _mask, _n_closest, _gamma_table),
    do: :erlang.nif_error(:nif_not_loaded)

  def kriging_objective(_model, _params, _lags, _sv, _weight),
    do: :erlang.nif_error(:nif_not_loaded)

  def kriging_statistics(_ref, _gamma_table), do: :erlang.nif_error(:nif_not_loaded)
  def kriging_distances(_ref, _targets, _mask), do: :erlang.nif_error(:nif_not_loaded)

  # Ordinary cokriging and the LMC fit.
  def cokriging_predict(_variables, _structures, _targets), do: :erlang.nif_error(:nif_not_loaded)

  def cokriging_fit_lmc(_n, _samples, _structures, _correct_diagonal),
    do: :erlang.nif_error(:nif_not_loaded)

  # Spatial random fields.
  def simulation_field(_model, _generator, _positions, _condition),
    do: :erlang.nif_error(:nif_not_loaded)
end
