Create a flow from YAML:
```yaml
id: random
namespace: company.team

tasks:
  - id: nr
    type: io.kestra.plugin.core.debug.Return
    format: "{{ randomInt(lower=0, upper=2) }}"

  - id: label
    type: io.kestra.plugin.core.execution.Labels
    labels:
      nr: "{{ outputs.nr.value }}"

  - id: log_data
    type: io.kestra.plugin.core.log.Log
    message: Hello there {{ outputs.nr.value }}

  - id: fail
    type: io.kestra.plugin.core.execution.Fail
    runIf: "{{ outputs.nr.value | number('INT') == 1 }}"
    errorMessage: Bad value returned!
```
Run this flow
Run this flow again

Replay the most recent execution of that flow
Replay the execution with id <first_execution_id>
Add labels replay:true and project:mcp to the execution <first_execution_id>