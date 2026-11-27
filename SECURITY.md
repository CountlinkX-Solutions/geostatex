# Security policy

## Supported versions

Security fixes go into the latest released minor version.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead: on the repository
page, **Security**, then **Report a vulnerability**. Say what is affected, how to reproduce it
and what an attacker gains.

You will get an answer within seven days. Once a fix is released, the advisory is published and
credits you unless you prefer otherwise.

Geostatex runs native code (a Rust NIF) inside the BEAM. Inputs that crash the VM, read or write
out of bounds, or exhaust memory despite the documented guards are in scope.
