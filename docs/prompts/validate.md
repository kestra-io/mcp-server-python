# Validate a flow before creating it

Use `validate_flow` before `create_flow_from_yaml` to catch syntax, type and
constraint errors while the flow can still be fixed for free.

- Pass the complete flow YAML as `source`.
- `{"valid": true, "errors": []}` means the flow is valid.
- `{"valid": false, "errors": [...], "message": ...}` lists what to fix.

If you have a whole namespace of files to check, use `validate_namespace_files`
with a map of `{"path.yaml": "source"}` entries.
