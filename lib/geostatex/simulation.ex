defmodule Geostatex.Simulation do
  @moduledoc """
  Gaussian spatial random fields by the randomisation method (`gstools.SRF`), unconditional or
  conditioned on data by kriging (`gstools.CondSRF`).

  A field is `mean + sqrt(psill / mode_no) * sum_j (z1_j cos(k_j . x) + z2_j sin(k_j . x))` plus
  white noise of variance `nugget`, with normal amplitudes `z1`, `z2` and wave vectors `k_j`
  drawn from the variogram model's spectral density: its covariance is the model's.

  The random numbers are GSTools' (numpy's legacy `RandomState`, a master stream seeding each
  use), so for 2D exponential and gaussian models a seed gives GSTools' field (verified to 1e-9).
  For 3D and spherical models GSTools samples the spectrum by MCMC; Geostatex samples it exactly
  instead, so the realisations differ from GSTools' for a seed but follow the same distribution.

  Models: `:spherical`, `:exponential` and `:gaussian` with `variogram_parameters`
  `%{psill:, range:, nugget:}` as in `Geostatex.Kriging`. GSTools' models map onto these with
  `rescale=1`: `Exponential(len_scale: range / 3)`, `Gaussian(len_scale: 4 / 7 * range)`,
  `Spherical(len_scale: range)`, `var: psill`, `anis: 1 / anisotropy_scaling`.

  Results are `%{field: values, seed: seed}`: `:points` gives one value per point, `:grid` rows of
  y as `Geostatex.Kriging.execute/5` (3D: planes of z), and `seed` is the seed used (`seed: nil`
  draws one).

  ## Examples

      iex> %{field: [a, b], seed: 7} =
      ...>   Geostatex.Simulation.unconditional(:points, [0.0, 50.0], [0.0, 0.0],
      ...>     variogram_model: :exponential,
      ...>     variogram_parameters: %{psill: 1.0, range: 30.0, nugget: 0.0},
      ...>     seed: 7
      ...>   )
      iex> is_float(a) and is_float(b)
      true
  """

  alias Geostatex.{Input, Kriging, Native}

  @models [:spherical, :exponential, :gaussian]
  @common [
    variogram_model: :spherical,
    variogram_parameters: nil,
    mean: 0.0,
    mode_no: 1000,
    seed: nil
  ]
  @defaults @common ++ [anisotropy_scaling: 1.0, anisotropy_angle: 0.0]
  @defaults_3d @common ++
                 [
                   anisotropy_scaling_y: 1.0,
                   anisotropy_scaling_z: 1.0,
                   anisotropy_angle_x: 0.0,
                   anisotropy_angle_y: 0.0,
                   anisotropy_angle_z: 0.0
                 ]
  @max_seed 4_294_967_295

  @doc """
  An unconditional 2D field (`gstools.SRF`) at `:points` (`{xs[i], ys[i]}`) or on the `:grid` of
  `xs` and `ys`. Options:

    * `:variogram_model` - `:spherical` (default), `:exponential` or `:gaussian`;
    * `:variogram_parameters` - `%{psill:, range:, nugget:}` (required);
    * `:mean` - the field's mean (default `0.0`);
    * `:mode_no` - the number of modes (default `1000`);
    * `:seed` - an integer in `0..4294967295`, or `nil` (default) to draw one;
    * `:anisotropy_scaling`, `:anisotropy_angle` - as in `Geostatex.Kriging.ordinary/3`,
      applied about the origin (GSTools' `anis` and `angles`).
  """
  @spec unconditional(:points | :grid, [number()], [number()], keyword()) :: map()
  def unconditional(style, xs, ys, opts \\ []) do
    opts = validate!(opts, @defaults)
    {positions, shape} = positions!(style, [xs, ys])
    anisotropy = {opts[:anisotropy_scaling] * 1.0, 1.0, 0.0, 0.0, opts[:anisotropy_angle] * 1.0}
    run(model_spec(opts, anisotropy), opts, positions, shape, nil)
  end

  @doc """
  An unconditional 3D field at `:points` or on the `:grid` of `xs`, `ys` and `zs`: the options of
  `unconditional/4` with the 3D anisotropy of `Geostatex.Kriging.ordinary_3d/3`.
  """
  @spec unconditional_3d(:points | :grid, [number()], [number()], [number()], keyword()) :: map()
  def unconditional_3d(style, xs, ys, zs, opts \\ []) do
    opts = validate!(opts, @defaults_3d)
    {positions, shape} = positions!(style, [xs, ys, zs])

    anisotropy =
      {opts[:anisotropy_scaling_y] * 1.0, opts[:anisotropy_scaling_z] * 1.0,
       opts[:anisotropy_angle_x] * 1.0, opts[:anisotropy_angle_y] * 1.0,
       opts[:anisotropy_angle_z] * 1.0}

    run(model_spec(opts, anisotropy), opts, positions, shape, nil)
  end

  @doc """
  A 2D field conditioned on the data of `kriging` (`gstools.CondSRF`), a simple or ordinary
  kriging model (`Geostatex.Kriging.simple/3`, `ordinary/3`) with a spherical, exponential or
  gaussian variogram: the kriging estimate plus the model's field scaled by the kriging standard
  deviation, `estimate + sqrt(max(v - nugget, 0) / psill) * field + sqrt(min(v, nugget) / nugget)
  * noise` (`v` the kriging variance). It reproduces the data where the kriging variance is 0
  (exact values; at a data point the variance is rounding noise, so the data are matched to
  about 1e-8, as in GSTools) and has the kriging estimate and variance as its pointwise mean and
  variance.
  The field uses the kriging's model and anisotropy. Options: `:mode_no`, `:seed`.
  """
  @spec conditional(Kriging.t(), :points | :grid, [number()], [number()], keyword()) :: map()
  def conditional(kriging, style, xs, ys, opts \\ [])

  def conditional(%Kriging{dim: 2} = k, style, xs, ys, opts) do
    {positions, shape} = positions!(style, [xs, ys])
    {pxs, pys} = positions |> Enum.map(&List.to_tuple/1) |> Enum.unzip()
    {z, s} = Kriging.execute(conditioning!(k), :points, pxs, pys)
    condition(k, opts, positions, shape, z, s)
  end

  def conditional(%Kriging{}, _style, _xs, _ys, _opts),
    do: raise(ArgumentError, "conditional/5 needs a 2D kriging model; use conditional_3d/6")

  @doc "A 3D field conditioned on a 3D simple or ordinary kriging model, as `conditional/5`."
  @spec conditional_3d(
          Kriging.t(),
          :points | :grid,
          [number()],
          [number()],
          [number()],
          keyword()
        ) ::
          map()
  def conditional_3d(kriging, style, xs, ys, zs, opts \\ [])

  def conditional_3d(%Kriging{dim: 3} = k, style, xs, ys, zs, opts) do
    {positions, shape} = positions!(style, [xs, ys, zs])
    [pxs, pys, pzs] = positions |> Enum.zip_with(& &1)
    {z, s} = Kriging.execute_3d(conditioning!(k), :points, pxs, pys, pzs)
    condition(k, opts, positions, shape, z, s)
  end

  def conditional_3d(%Kriging{}, _style, _xs, _ys, _zs, _opts),
    do: raise(ArgumentError, "conditional_3d/6 needs a 3D kriging model; use conditional/5")

  defp condition(k, opts, positions, shape, z, s) do
    opts =
      validate!(
        opts,
        [mode_no: 1000, seed: nil],
        variogram_model: k.variogram_model,
        variogram_parameters: k.variogram_model_parameters,
        mean: 0.0
      )

    bin = fn v -> v |> Input.to_f64_binary() |> elem(0) end
    run(model_spec(opts, k.anisotropy), opts, positions, shape, {bin.(z), bin.(s)})
  end

  defp conditioning!(%Kriging{variogram_model: model, drift_terms: []} = k)
       when model in @models,
       do: k

  defp conditioning!(_k),
    do:
      raise(
        ArgumentError,
        "conditioning needs a simple or ordinary kriging model with a spherical, exponential " <>
          "or gaussian variogram"
      )

  defp run({model, params, anisotropy}, opts, positions, shape, condition) do
    dim = positions |> List.first([0, 0]) |> length()
    {bin, _} = positions |> List.flatten() |> Input.to_f64_binary()

    field =
      {model, params, anisotropy}
      |> Native.simulation_field(
        {opts[:mode_no], opts[:seed], opts[:mean] * 1.0},
        {bin, dim},
        condition
      )
      |> unwrap()
      |> Input.from_f64_binary()

    %{field: reshape(field, shape), seed: opts[:seed]}
  end

  defp model_spec(opts, anisotropy) do
    {params, _} = Input.to_f64_binary(opts[:variogram_parameters])
    {Atom.to_string(opts[:variogram_model]), params, anisotropy}
  end

  # GSTools' order: points as given, grids x-major (x slowest, as numpy's `ij` mesh), which the
  # nugget noise follows; the shape tells how to lay a grid out as Kriging does.
  defp positions!(:points, axes) do
    unless Enum.all?(axes, &is_list/1) and
             axes |> Enum.map(&length/1) |> Enum.uniq() |> length() == 1,
           do: raise(ArgumentError, ":points needs coordinate lists of the same length")

    {axes |> Enum.zip_with(& &1) |> Enum.map(&floats/1), :flat}
  end

  defp positions!(:grid, axes) do
    unless Enum.all?(axes, &(is_list(&1) and &1 != [])),
      do: raise(ArgumentError, ":grid needs a non-empty list for each axis")

    positions =
      Enum.reduce(Enum.reverse(axes), [[]], fn axis, acc ->
        for v <- axis, rest <- acc, do: [v | rest]
      end)

    {Enum.map(positions, &floats/1), {:grid, Enum.map(axes, &length/1)}}
  end

  defp positions!(style, _axes),
    do: raise(ArgumentError, "style must be :points or :grid, got: #{inspect(style)}")

  defp floats(point), do: Enum.map(point, &(&1 * 1.0))

  defp reshape(field, :flat), do: field

  # x-major values to rows of y (2D) or planes of z of rows of y (3D).
  defp reshape(field, {:grid, [nx, ny]}),
    do: rows(List.to_tuple(field), nx, ny, &(&1 * ny + &2))

  defp reshape(field, {:grid, [nx, ny, nz]}) do
    t = List.to_tuple(field)
    for iz <- 0..(nz - 1), do: rows(t, nx, ny, &((&1 * ny + &2) * nz + iz))
  end

  # Rows of y of the values at `index.(ix, iy)`.
  defp rows(t, nx, ny, index) do
    for iy <- 0..(ny - 1), do: for(ix <- 0..(nx - 1), do: elem(t, index.(ix, iy)))
  end

  defp validate!(opts, defaults, fixed \\ []) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- Keyword.keys(defaults) do
      [] -> :ok
      unknown -> raise ArgumentError, "unknown option(s) #{inspect(unknown)}"
    end

    opts = defaults |> Keyword.merge(opts) |> Keyword.merge(fixed) |> Map.new()
    model!(opts[:variogram_model])
    mode_no!(opts[:mode_no])
    unless is_number(opts[:mean]), do: raise(ArgumentError, ":mean must be a number")

    %{
      opts
      | seed: seed!(opts[:seed]),
        variogram_parameters: parameters!(opts[:variogram_parameters])
    }
  end

  defp model!(model) when model in @models, do: :ok

  defp model!(_model),
    do: raise(ArgumentError, ":variogram_model must be :spherical, exponential or gaussian")

  defp mode_no!(n) when is_integer(n) and n >= 1, do: :ok
  defp mode_no!(_n), do: raise(ArgumentError, ":mode_no must be a positive integer")

  defp seed!(nil), do: :rand.uniform(@max_seed + 1) - 1
  defp seed!(s) when is_integer(s) and s >= 0 and s <= @max_seed, do: s

  defp seed!(other),
    do:
      raise(ArgumentError, ":seed must be an integer in 0..#{@max_seed}, got: #{inspect(other)}")

  defp parameters!(%{psill: p, range: r} = m) when is_number(p) and is_number(r),
    do: [p, r, Map.get(m, :nugget, 0.0)]

  defp parameters!([p, r, n]) when is_number(p) and is_number(r) and is_number(n), do: [p, r, n]

  defp parameters!(other),
    do:
      raise(
        ArgumentError,
        ":variogram_parameters must be %{psill:, range:, nugget:}, got: #{inspect(other)}"
      )

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
