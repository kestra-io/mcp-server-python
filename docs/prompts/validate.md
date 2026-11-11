# Validate a flow before creating it

Use `validate_flow` before `create_flow_from_yaml` to catch syntax, type and
constraint errors while the flow can still be fixed for free.

- Pass the complete flow YAML as `source`. Several flows can be separated by `---`.
- The result is `{"valid": ..., "documents": [...]}` with one entry per flow.
  `valid` is true only when every flow is valid.
- An invalid entry has `valid: false`, an `errors` list (`detail`, plus
  `pointer` and `path` on Kestra versions that report them) and the raw
  `constraints` text. Fix what they point to and validate again.
- `warnings`, `deprecationPaths`, `infos` and `outdated` are not errors, but
  are worth reading before creating the flow.

If you have a whole namespace of files to check, use `validate_namespace_files`
with a map of `{"path.yaml": "source"}` entries. Each file gets the same result
shape, plus an overall `valid` flag.
