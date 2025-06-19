Create a flow:
id: mcp
namespace: dev
concurrency:
  behavior: QUEUE
  limit: 1
tasks:
  - id: greet
    type: io.kestra.plugin.core.log.Log
    message: Hello from the API!
  - id: sleep
    type: io.kestra.plugin.core.flow.Sleep
    duration: PT15M
  - id: wake_up
    type: io.kestra.plugin.core.log.Log
    message: Wake up!

Run this flow
Run this flow again

Force run an execution for <second_execution>
Pause the execution with id <second_execution>
Resume the execution with id <second_execution>

Create a flow from YAML:
id: pause_demo
namespace: dev
tasks:
  - id: hi
    type: io.kestra.plugin.core.log.Log
    message: Hello World!
  - id: pause
    type: io.kestra.plugin.core.flow.Pause
  - id: welcome_back
    type: io.kestra.plugin.core.log.Log
    message: Welcome back!

Trigger an execution of that flow
Trigger an execution of that flow again
Resume the execution with id <first_execution_id>
Resume the most recent paused execution for that flow
Resume the most recent paused execution for the flow pause_demo in the namespace dev

Create a flow from YAML:
```yaml
id: approval
namespace: company

inputs:
  - id: request
    type: STRING
    defaults: John Doe requests PTO for last 2 weeks of July 2025

tasks:
  - id: waitForApproval
    type: io.kestra.plugin.core.flow.Pause
    onResume:
      - id: approved
        description: Approve the request?
        type: BOOLEAN # use BOOL in kestra 0.23+ and BOOLEAN in kestra <= 0.22.9
        defaults: true
        
      - id: reason
        description: Reason for approval or rejection?
        type: STRING
        defaults: Approved

  - id: approve
    type: io.kestra.plugin.core.http.Request
    uri: https://kestra.io/api/mock
    method: POST
    contentType: application/json
    body: "{{ outputs.waitForApproval.onResume }}"

  - id: log
    type: io.kestra.plugin.core.log.Log
    message: Status is {{ outputs.waitForApproval.onResume.reason }}
```    

Trigger an execution of that flow
Trigger an execution of that flow again
Resume latest execution of that flow
Trigger an execution of the flow approval in the namespace company.team
Resume the execution with id <first_execution_id> with inputs reason=no more PTO available
