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

Backfill executions for the flow scheduled_flow in the namespace dev for the last 3 hours
Backfill executions for the flow scheduled_flow in the namespace dev for the last 2 days
