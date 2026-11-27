defmodule Geostatex.MixProject do
  use Mix.Project

  @version "0.5.0"
  @source_url "https://github.com/CountlinkX-Solutions/geostatex"

  def project do
    [
      app: :geostatex,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Geostatistics for Elixir: variograms, kriging and Gaussian random fields, backed by a Rust core.",
      name: "Geostatex",
      source_url: @source_url,
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp docs do
    [
      main: "readme",
      # "View source" links point at the release tag, so published docs match the published code.
      source_ref: "v#{@version}",
      extras: [
        "README.md",
        "CHANGELOG.md": [title: "Changelog"],
        "notebooks/meuse_case.livemd": [title: "Use case: heavy metals in the Meuse floodplain"],
        "notebooks/es/caso_mosa.livemd": [title: "Caso de uso: metales pesados en el Mosa (es)"],
        "notebooks/baltimore_case.livemd": [title: "Use case: house prices in Baltimore"],
        "notebooks/es/caso_precios_baltimore.livemd": [
          title: "Caso de uso: precios en Baltimore (es)"
        ],
        "notebooks/us_income_case.livemd": [title: "Use case: US state income in space and time"],
        "notebooks/es/caso_renta_eeuu.livemd": [title: "Caso de uso: renta de EE. UU. (es)"],
        "notebooks/baltimore_ml_case.livemd": [title: "Use case: machine learning and kriging"],
        "notebooks/es/caso_ml_baltimore.livemd": [
          title: "Caso de uso: aprendizaje automático (es)"
        ],
        "notebooks/baltimore_xgboost_case.livemd": [title: "Use case: XGBoost and kriging"],
        "notebooks/es/caso_xgboost_baltimore.livemd": [
          title: "Caso de uso: XGBoost y kriging (es)"
        ],
        "notebooks/baltimore_satellite_case.livemd": [
          title: "Use case: aerial images as features"
        ],
        "notebooks/es/caso_satelite_baltimore.livemd": [
          title: "Caso de uso: imágenes aéreas (es)"
        ],
        "notebooks/king_county_images_case.livemd": [
          title: "Use case: aerial images in King County"
        ],
        "docs/architecture.md": [title: "Architecture"],
        "docs/parity.md": [title: "Parity"],
        "CONTRIBUTING.md": [title: "Contributing"],
        LICENSE: [title: "License (LGPL-3.0)"],
        COPYING: [title: "GNU General Public License"],
        NOTICE: [title: "Notice"]
      ],
      # Geostatex.Native is internal (@moduledoc false) but the architecture doc names it.
      skip_undefined_reference_warnings_on: ["docs/architecture.md"],
      groups_for_extras: [
        Guides: ~r{docs/},
        Notebooks: ~r{notebooks/},
        Legal: ["LICENSE", "COPYING", "NOTICE"]
      ]
    ]
  end

  # Ship the NIF sources only; `priv/native` is a build output and never packaged.
  defp package do
    [
      licenses: ["LGPL-3.0-only"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md"
      },
      files:
        ~w(lib native/geostatex_core/src native/geostatex_core/Cargo.toml
           native/geostatex_native/src native/geostatex_native/Cargo.toml
           .cargo/config.toml Cargo.toml Cargo.lock mix.exs docs README.md CHANGELOG.md LICENSE COPYING LICENSES NOTICE) ++
          checksum_files()
    ]
  end

  # The checksum file of the precompiled NIFs only exists once a release was built (README,
  # "Releasing"); with it in the package, Geostatex.Native downloads instead of building. Hex
  # rejects an entry that matches nothing, so it is listed only when it exists.
  @doc false
  def checksum_files(root \\ __DIR__) do
    root
    |> Path.join("checksum-*.exs")
    |> Path.wildcard()
    |> Enum.map(&Path.basename/1)
    |> Enum.sort()
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Only a source build needs it (Geostatex.Native): optional for users of the precompiled NIF.
      # The crate is rustler 0.39; Hex has the Mix package 0.38.0, which builds it.
      {:rustler, "~> 0.38", runtime: false, optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 0.10 or ~> 1.0", optional: true},
      {:explorer, "~> 0.13", optional: true},
      {:stream_data, "~> 1.4", only: [:dev, :test]},
      # Only the oracle helper (test/support) decodes fixtures with it. Rustler needs it in every
      # environment, so it cannot be test-only here; optional keeps it away from users.
      {:jason, "~> 1.4", optional: true},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40", only: [:dev, :test], runtime: false},
      # Geostatex.GeoFrame calls GeoFrame at runtime only; the dependency exists to test it.
      {:geoframe, "~> 0.3.3", only: :test}
    ]
  end
end
