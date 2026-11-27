## What and why

## How it was verified

- [ ] `mix test` and `cargo test --workspace` pass
- [ ] `mix format --check-formatted`, `mix credo --strict`, `cargo fmt --check` and
      `cargo clippy --workspace --all-targets -- -D warnings` are clean
- [ ] New behaviour has a test; results that follow a reference library have a fixture
- [ ] `CHANGELOG.md` and `docs/parity.md` are updated when users can notice the change
- [ ] I did not read GPL-licensed source code to write this change (see CONTRIBUTING.md)
