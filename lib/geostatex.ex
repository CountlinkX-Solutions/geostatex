defmodule Geostatex do
  @moduledoc """
  Geostatistics for Elixir: experimental variograms, theoretical variogram models and kriging,
  after scikit-gstat and PyKrige, with a pure-Rust core behind a thin NIF layer.

  Results are verified against fixtures generated with the reference libraries (see
  `docs/parity.md`). Inputs are lists of numbers or points, `Nx` tensors or `Explorer` series
  (see `Geostatex.Input`).
  """
end
