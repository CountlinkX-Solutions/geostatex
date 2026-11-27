defmodule Geostatex.Cokriging do
  @moduledoc """
  Ordinary cokriging with a linear model of coregionalisation (R `gstat`'s multivariable
  prediction, each variable with a constant mean `~ 1`), and the model's fit (`gstat::fit.lmc`).

  A linear model of coregionalisation (LMC) of `n` variables is a list of structures, each a
  model, a range and a symmetric, positive semi-definite `n x n` matrix of partial sills: every
  direct and cross variogram is `sum sills[a][b] * unit(h)`. Models: `:nugget`, `:spherical`,
  `:exponential` and `:gaussian` with Geostatex's (PyKrige's) ranges (gstat's `Exp` and `Gau`
  take `range / 3` and `4 / 7 * range`).

  The variables may be observed at different points (heterotopic data). The system holds the
  covariances between every observation, bordered by one unbiasedness constraint per variable;
  `execute/5` gives each variable's estimate, its cokriging variance and the error covariances
  between variables.

  ## Examples

      iex> zn = {[{0, 0}, {10, 0}, {0, 10}, {10, 10}], [1.0, 2.0, 1.5, 3.0]}
      iex> cu = {[{0, 0}, {10, 0}, {0, 10}, {10, 10}], [10.0, 12.0, 11.0, 14.0]}
      iex> ck = Geostatex.Cokriging.new([zn, cu],
      ...>   structures: [
      ...>     %{model: :nugget, sills: [[0.05, 0.02], [0.02, 0.1]]},
      ...>     %{model: :spherical, range: 20.0, sills: [[0.5, 0.6], [0.6, 1.2]]}
      ...>   ])
      iex> %{estimates: [[z1], [z2]]} = Geostatex.Cokriging.execute(ck, :points, [0.0], [0.0])
      iex> {Float.round(z1, 9), Float.round(z2, 9)}
      {1.0, 10.0}
  """

  alias Geostatex.{Input, Native}

  @enforce_keys [:variables, :structures, :n]
  defstruct @enforce_keys

  @type t :: %__MODULE__{}

  @models [:nugget, :spherical, :exponential, :gaussian]

  @doc """
  Sets up ordinary cokriging of `variables`, a list of `{points, values}` (2D points), with the
  LMC `structures:` (a list of `%{model:, range:, sills:}`; `:nugget` needs no range). Raises if
  the model is not a valid LMC.
  """
  @spec new([{term(), term()}], keyword()) :: t()
  def new(variables, opts) do
    unless is_list(variables) and variables != [] and
             Enum.all?(variables, &match?({_, _}, &1)),
           do: raise(ArgumentError, "variables must be a non-empty list of {points, values}")

    unless Keyword.keyword?(opts) and Keyword.keys(opts) == [:structures],
      do: raise(ArgumentError, "the only option is :structures")

    n = length(variables)

    variables =
      Enum.map(variables, fn {points, values} ->
        {bin, count} = Input.to_points_binary(points)
        {vals, _} = Input.to_f64_binary(values, name: "values", length: count)
        {bin, vals}
      end)

    ck = %__MODULE__{variables: variables, structures: structures!(opts[:structures], n), n: n}
    predict(ck, [], :flat)
    ck
  end

  @doc """
  Predictions at `:points` (`{xs[i], ys[i]}`) or on the `:grid` of `xs` and `ys` (rows of y):
  `%{estimates:, variances:, covariances:}`, with one list per variable for the estimates and
  variances and `covariances` keyed by `{i, j}` (`i < j`, 0-based).
  """
  @spec execute(t(), :points | :grid, [number()], [number()], keyword()) :: map()
  def execute(%__MODULE__{} = ck, style, xs, ys, opts \\ []) do
    if opts != [], do: raise(ArgumentError, "execute/5 takes no options")

    {targets, shape} =
      case style do
        :points when is_list(xs) and is_list(ys) and length(xs) == length(ys) ->
          {Enum.zip(xs, ys), :flat}

        :grid when is_list(xs) and is_list(ys) ->
          {for(y <- ys, x <- xs, do: {x, y}), {length(ys), length(xs)}}

        _ ->
          raise ArgumentError, "style must be :points (lists of the same length) or :grid"
      end

    predict(ck, targets, shape)
  end

  defp predict(ck, targets, shape) do
    {bin, _} = if targets == [], do: {<<>>, 0}, else: Input.to_points_binary(targets)

    {estimates, covariances} =
      ck.variables |> Native.cokriging_predict(ck.structures, {bin, 2}) |> unwrap()

    n = ck.n
    estimates = Enum.map(estimates, &reshape(Input.from_f64_binary(&1), shape))
    matrices = covariances |> Input.from_f64_binary() |> Enum.chunk_every(n * n)
    at = fn i, j -> matrices |> Enum.map(&Enum.at(&1, i * n + j)) |> reshape(shape) end

    %{
      estimates: estimates,
      variances: for(i <- 0..(n - 1), do: at.(i, i)),
      covariances: Map.new(for i <- 0..(n - 1), j <- (i + 1)..(n - 1)//1, do: {{i, j}, at.(i, j)})
    }
  end

  @doc """
  gstat's `fit.lmc` with the ranges fixed: fits the partial sills of `structures` (a list of
  `%{model:, range:}`) to the sample `samples`, a map from `{i, j}` (`i <= j`, 0-based, every
  pair once) to `%{lags:, gamma:, counts:}` (as gstat's `variogram()` gives them: `dist`,
  `gamma`, `np`). Each direct and cross variogram is fitted by weighted least squares (weights
  `counts / lags^2`, gstat's `fit.method = 7`); each structure's sill matrix is then replaced by
  its closest positive semi-definite matrix and its diagonal multiplied by
  `:correct_diagonal` (default `1.0`). Returns the structures with their `sills:`.
  """
  @spec fit_lmc(map(), [map()], keyword()) :: [map()]
  def fit_lmc(samples, structures, opts \\ []) do
    correct = Keyword.get(opts, :correct_diagonal, 1.0)

    unless is_map(samples) and samples != [],
      do: raise(ArgumentError, "samples must be a map from {i, j} to sample variograms")

    n = samples |> Map.keys() |> Enum.flat_map(&Tuple.to_list/1) |> Enum.max() |> Kernel.+(1)

    unless map_size(samples) == div(n * (n + 1), 2),
      do: raise(ArgumentError, "samples need one per pair {i, j} with i <= j (#{n} variables)")

    bin = fn v -> v |> Input.to_f64_binary() |> elem(0) end

    wire =
      for {{i, j}, s} <- samples,
          do: {i, j, bin.(s.lags), bin.(s.gamma), bin.(s.counts)}

    kinds = for s <- structures, do: {Atom.to_string(model!(s)), range!(s)}

    wire
    |> then(&Native.cokriging_fit_lmc(n, &1, kinds, correct * 1.0))
    |> unwrap()
    |> Enum.map(fn {model, range, sills} ->
      %{
        model: String.to_existing_atom(model),
        range: range,
        sills: sills |> Input.from_f64_binary() |> Enum.chunk_every(n)
      }
    end)
  end

  defp structures!(structures, n) when is_list(structures) and structures != [] do
    for s <- structures do
      sills = Map.get(s, :sills)

      unless is_list(sills) and length(sills) == n and
               Enum.all?(sills, &(is_list(&1) and length(&1) == n)),
             do: raise(ArgumentError, "every structure needs :sills, a #{n} x #{n} matrix")

      {Atom.to_string(model!(s)), range!(s),
       sills |> List.flatten() |> Input.to_f64_binary() |> elem(0)}
    end
  end

  defp structures!(other, _n),
    do: raise(ArgumentError, ":structures must be a non-empty list, got: #{inspect(other)}")

  defp model!(%{model: model}) when model in @models, do: model

  defp model!(s),
    do:
      raise(
        ArgumentError,
        "every structure needs a :model in #{inspect(@models)}, got: #{inspect(s)}"
      )

  defp range!(%{model: :nugget}), do: 0.0
  defp range!(%{range: r}) when is_number(r) and r > 0, do: r * 1.0

  defp range!(s),
    do:
      raise(
        ArgumentError,
        "every structure but :nugget needs a positive :range, got: #{inspect(s)}"
      )

  defp reshape(values, :flat), do: values
  defp reshape(values, {_ny, nx}), do: Enum.chunk_every(values, nx)

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
