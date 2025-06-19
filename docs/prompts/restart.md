Create a flow from YAML:
```yaml
id: failure
namespace: company.team

tasks:
  - id: start
    type: io.kestra.plugin.core.log.Log
    message: Hello World!

  - id: fail
    type: io.kestra.plugin.core.execution.Fail

  - id: run_after_restart
    type: io.kestra.plugin.core.log.Log
    message: Hello after restart!
```
Run this flow
Run this flow again

Restart the most recent execution of that flow
Restart the execution with id <first_execution_id>
Restart the execution with id <first_execution_id> from a failed task

Set the taskrun of the task fail to WARNING in the execution <first_execution_id>
Change the execution status to SUCCESS for <first_execution_id>
