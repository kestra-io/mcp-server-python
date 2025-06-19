Get all flow dependencies for the flow called flow3 in the namespace company
Disable flow hello-world from namespace tutorial
Enable flow hello-world from namespace tutorial
Get the YAML definition of the flow hello-world in the namespace tutorial

Find all flows containing automation
Find all flows containing Hello World
Seach for flows including Hello World in its source code
Get all revisions for the flow hello-world in namespace tutorial
Get all flow dependencies for the flow hello-world in the namespace tutorial
Find flows that have triggers
Find flows that have triggers and return detailed information about the trigger type, whether it's enabled etc

List flows with active triggers
List flows with disabled triggers

Create a flow with YAML:
```yaml
id: scheduled_flow  
namespace: dev  
tasks:  
  - id: label  
    type: io.kestra.plugin.core.execution.Labels  
    labels:  
      scheduledDate: "{{trigger.date ?? execution.startDate}}"  
  - id: external_system_export  
    type: io.kestra.plugin.scripts.shell.Commands  
    taskRunner:  
      type: io.kestra.plugin.core.runner.Process  
    commands:  
      - echo "processing data for {{trigger.date ?? execution.startDate}}"  
      - sleep $((RANDOM % 5 + 1))  
triggers:  
  - id: schedule  
    type: io.kestra.plugin.core.trigger.Schedule  
    cron: "*/30 * * * *"  
```

Create a flow from YAML:
id: sleep
namespace: company.team

tasks:
  - id: sleep
    type: io.kestra.plugin.core.flow.Sleep
    duration: PT15M


Update the flow as follows:
id: hello-world
namespace: tutorial
description: Hello World

inputs:
  - id: user
    type: STRING
    defaults: Rick Astley

tasks:
  - id: first_task
    type: io.kestra.plugin.core.debug.Return
    format: thrilled

  - id: hello_world
    type: io.kestra.plugin.core.log.Log
    message: Welcome to Kestra, {{inputs.user}}! We are {{outputs.first_task.value}} to have You here!
    
  - id: subflow
    type: io.kestra.plugin.core.flow.Subflow
    namespace: tutorial
    flowId: business-automation


