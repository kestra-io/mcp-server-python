Below are example prompts and example response each of them generated

## Core
Create a flow that sends a POST request to https://dummyjson.com/products/add
id: send_post_request
namespace: dev

tasks:
  - id: post_to_dummyjson
    type: io.kestra.plugin.core.http.Request
    uri: https://dummyjson.com/products/add
    method: POST
    contentType: application/json
    body: |
      {
        "title": "Kestra Test Product",
        "price": 100
      }

Create a flow that logs Hello RAG to the console
id: hello_rag_flow
namespace: dev.rag

tasks:
  - id: log_hello
    type: io.kestra.plugin.core.log.Log
    message: "Hello RAG"

Create a flow that returns Hello RAG as output
id: hello_rag_flow
namespace: dev

tasks:
  - id: return_hello
    type: io.kestra.plugin.core.debug.Return
    format: "Hello RAG"

Create a flow that outputs Hello RAG as value
id: output_hello_rag
namespace: company.team

tasks:
  - id: output_message
    type: io.kestra.plugin.core.output.OutputValues
    values:
      message: "Hello RAG"

Run a flow every 10 minutes
id: scheduled_every_10_minutes
namespace: dev

tasks:
  - id: log_schedule_time
    type: io.kestra.plugin.core.log.Log
    message: "Flow triggered at {{ trigger.date ?? execution.startDate }}"

triggers:
  - id: every_10_minutes
    type: io.kestra.plugin.core.trigger.Schedule
    cron: "*/10 * * * *"

Run a flow every day at 9 AM
id: daily_flow_at_9am
namespace: company.team

tasks:
  - id: log_schedule_time
    type: io.kestra.plugin.core.log.Log
    message: Flow triggered at {{ trigger.date }}

triggers:
  - id: schedule_daily_9am
    type: io.kestra.plugin.core.trigger.Schedule
    cron: "0 9 * * *"


Make a REST API call to https://kestra.io/api/mock and allow failure
id: http_request_allow_failure
namespace: company.team

tasks:
  - id: call_mock_api
    type: io.kestra.plugin.core.http.Request
    uri: https://kestra.io/api/mock
    options:
      allowFailed: true

## Docker

Run a command echo 'Hello Docker' in a Docker container
id: run_echo_in_docker
namespace: dev

tasks:
  - id: run_command
    type: io.kestra.plugin.docker.Run
    containerImage: ubuntu
    commands:
      - echo
      - Hello Docker


Run a command python main.py in a Docker container
id: run_python_script
namespace: company.team

tasks:
  - id: run_main_py
    type: io.kestra.plugin.scripts.python.Commands
    commands:
      - python main.py


Run a script main.py stored as namespace file
id: run-namespace-python-script
namespace: company.team

tasks:
  - id: run-main-script
    type: io.kestra.plugin.scripts.python.Commands
    namespaceFiles:
      enabled: true
    commands:
      - python main.py


Build a Docker image from an inline Dockerfile and push it to a GitHub Container Registry
id: build_and_push_docker_ghcr
namespace: dev

tasks:
  - id: build_image_and_push
    type: io.kestra.plugin.docker.Build
    dockerfile: |
      FROM ubuntu
      RUN echo "Building image for Kestra"
      CMD ["echo", "Image built successfully"]
    tags:
      - ghcr.io/your-github-username/your-image-name:latest
    push: true
    credentials:
      username: your-github-username
      password: "{{ secret('GITHUB_TOKEN') }}"


Build a Docker image from an inline Dockerfile and push it to a DockerHub Container Registry
id: docker_build_push_dockerhub
namespace: dev

tasks:
  - id: build_and_push
    type: io.kestra.plugin.docker.Build
    dockerfile: |
      FROM ubuntu:latest
      RUN echo "Hello from Dockerfile!"
    tags:
      - your_dockerhub_username/your_image_name:latest # Replace with your DockerHub username and desired image name/tag
    push: true
    credentials:
      registry: https://index.docker.io/v1/
      username: "{{ secret('DOCKERHUB_USERNAME') }}"
      password: "{{ secret('DOCKERHUB_PASSWORD') }}"


Shell
Run a bash script
Run a Shell command that... 

## KV
Create a flow that adds a string KV pair called MYKEY with value myvalue to namespace company
id: set_mykey_in_company
namespace: company.team

tasks:
  - id: set_kv_pair
    type: io.kestra.plugin.core.kv.Set
    key: MYKEY
    value: myvalue
    namespace: company
    overwrite: true
    kvType: STRING

Fetch value for KV pair called MYKEY from namespace company
id: fetch_mykey_value
namespace: company

tasks:
  - id: get_mykey
    type: io.kestra.plugin.core.kv.Get
    key: MYKEY
    namespace: company

## S3
Create a flow that downloads a file mydata.csv from S3 bucket named mybucket
id: download_s3_file
namespace: company.team

tasks:
  - id: download_mydata_csv
    type: io.kestra.plugin.aws.s3.Download
    bucket: mybucket
    key: mydata.csv

Create a flow that downloads all files from the folder kestra/plugins/ from S3 bucket mybucket in us-east-1
id: download_s3_folder_files
namespace: company.team

tasks:
  - id: download_kestra_plugins
    type: io.kestra.plugin.aws.s3.Downloads
    bucket: mybucket
    region: us-east-1
    prefix: kestra/plugins/
    filter: FILES
    action: NONE

## Slack
Send a Slack notification that approval is needed and Pause the flow for manual approval
id: approval_request_flow
namespace: dev

inputs:
  - id: slack_webhook_url
    type: URI
    description: The Slack incoming webhook URL to send the notification.

tasks:
  - id: send_approval_notification
    type: io.kestra.plugin.notifications.slack.SlackIncomingWebhook
    url: "{{ inputs.slack_webhook_url }}"
    payload: |
      {
        "text": "Approval is needed for Kestra flow `{{ flow.namespace }}.{{ flow.id }}` execution `{{ execution.id }}`. Please review and resume the execution here: http://localhost:8080/ui/executions/{{ flow.namespace }}/{{ flow.id }}/{{ execution.id }}"
      }

  - id: pause_for_manual_approval
    type: io.kestra.plugin.core.flow.Pause
    description: Waiting for manual approval via the Kestra UI.


Send a Slack alert whenever any execution from namespace company fails
id: failed_company_alert
namespace: system

tasks:
  - id: send_slack_alert
    type: io.kestra.plugin.notifications.slack.SlackExecution
    url: "{{ secret('SLACK_WEBHOOK') }}"
    channel: "#alerts"
    executionId: "{{ trigger.executionId }}"

triggers:
  - id: on_company_failure
    type: io.kestra.plugin.core.trigger.Flow
    conditions:
      - type: io.kestra.plugin.core.condition.ExecutionStatus
        in:
          - FAILED
      - type: io.kestra.plugin.core.condition.ExecutionNamespace
        namespace: company
        prefix: true

## HTTP, Python, Node.js
Send a POST request to https://dummyjson.com/products/add
id: send_dummyjson_post_request
namespace: dev

tasks:
  - id: send_post_request
    type: io.kestra.plugin.core.http.Request
    uri: https://dummyjson.com/products/add
    method: POST
    contentType: application/json
    body: |
      {
        "title": "Kestra Test Product"
      }

Write a Python script that sends a POST request to https://dummyjson.com/products/add
id: python_post_request
namespace: company.team

tasks:
  - id: send_post_request
    type: io.kestra.plugin.scripts.python.Script
    beforeCommands:
      - pip install requests
    script: |
      import requests
      import json

      url = "https://dummyjson.com/products/add"
      payload = {
          "title": "Kestra Automated Product",
          "description": "This product was added using a Kestra flow.",
          "price": 50,
          "brand": "Kestra",
          "category": "automation-tools"
      }

      headers = {
          "Content-Type": "application/json"
      }

      response = requests.post(url, data=json.dumps(payload), headers=headers)

      print(f"Status Code: {response.status_code}")
      print("Response Body:")
      print(response.text)

      # Optional: Output the response body or status code
      # from kestra import Kestra
      # Kestra.outputs({'status_code': response.status_code, 'response_body': response.json()})


Write a Node.js script that sends a POST request to https://dummyjson.com/products/add
id: nodejs_post_request
namespace: company.team

tasks:
  - id: send_post_request_nodejs
    type: io.kestra.plugin.scripts.node.Script
    warningOnStdErr: false
    beforeCommands:
      - npm install node-fetch@2
    script: |\
      const fetch = require('node-fetch');

      async function postData() {
        const url = 'https://dummyjson.com/products/add';
        const data = {
          title: 'Kestra Test Product',
          description: 'This is a test product added via Kestra Node.js script.',
          price: 999,
          brand: 'Kestra'
        };

        try {
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
          });

          console.log(`Status Code: ${response.status}`);
          const responseData = await response.json();
          console.log('Response Body:', JSON.stringify(responseData, null, 2));

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }\

        } catch (error) {
          console.error('Error sending POST request:', error);
          throw error;
        }
      }

      postData();

Create a flow with a Python script that fetches weather data for New York City
id: fetch-nyc-weather
namespace: company.team

tasks:
  - id: fetch_weather_data
    type: io.kestra.plugin.scripts.python.Script
    description: Fetch weather data for New York City using a Python script
    beforeCommands:
      - pip install requests
    script: |
      import requests
      import json
      import os

      city = "New York"
      country_code = "US"
      location = f"{city},{country_code}"

      # Replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual API key,
      # or get it from Kestra Secrets (e.g., os.environ.get('OPENWEATHERMAP_API_KEY'))
      api_key = "YOUR_OPENWEATHERMAP_API_KEY"
      base_url = "http://api.openweathermap.org/data/2.5/weather?"

      if api_key == "YOUR_OPENWEATHERMAP_API_KEY":
          print("WARNING: Please replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual API key.")
          print("Skipping API call and printing mock data.")
          weather_data = {"city": city, "temperature": "20C (Mock)", "conditions": "Sunny (Mock)"}
          print(json.dumps(weather_data, indent=2))
      else:
          complete_url = f"{base_url}appid={api_key}&q={location}"
          print(f"Fetching weather for {city}...")
          try:
              response = requests.get(complete_url)
              response.raise_for_status()
              weather_data = response.json()
              print(f"Weather data for {city}:")
              print(json.dumps(weather_data, indent=2))
          except requests.exceptions.RequestException as e:
              print(f"Error fetching weather data: {e}")
              # raise e


## Redis

Fetch value for string kv pair called mykey from Redis
id: fetch_mykey_from_redis
namespace: dev.kestra.redis

tasks:
  - id: get_mykey_value
    type: io.kestra.plugin.redis.string.Get
    url: redis://localhost:6379/0
    key: mykey
    serdeType: STRING


Fetch value for mykey from Redis
id: fetch_redis_key
namespace: company.team

tasks:
  - id: get_mykey
    type: io.kestra.plugin.redis.string.Get
    url: redis://:redis@localhost:6379/0
    key: mykey

Set value for mykey in Redis to myvalue
id: set_redis_key
namespace: company.team

tasks:
  - id: set
    type: io.kestra.plugin.redis.string.Set
    url: redis://localhost:6379/0
    key: mykey
    value: myvalue
    serdeType: STRING

## dbt

Create a flow that runs a dbt build command on DuckDB
id: dbt_build_duckdb
namespace: company.team

tasks:
  - id: working_directory
    type: io.kestra.plugin.core.flow.WorkingDirectory
    tasks:
      - id: clone_repository
        type: io.kestra.plugin.git.Clone
        url: https://github.com/kestra-io/dbt-demo
        branch: main

      - id: dbt_build
        type: io.kestra.plugin.dbt.cli.Build
        taskRunner:
          type: io.kestra.plugin.scripts.runner.docker.Docker
        dbtPath: /usr/local/bin/dbt
        containerImage: ghcr.io/kestra-io/dbt-duckdb
        profiles: |
          jaffle_shop:
            outputs:
              dev:
                type: duckdb
                path: ':memory:'
                extensions:
                  - parquet
            target: dev


## Git
Sync all flows and scripts for selected namespaces from Git to Kestra
id: git_sync
namespace: system

tasks:
  - id: sync
    type: io.kestra.plugin.core.flow.ForEach
    values: ["company", "company.team", "company.analytics"]
    tasks:
      - id: flows
        type: io.kestra.plugin.git.SyncFlows
        targetNamespace: "{{ taskrun.value }}"
        gitDirectory: "{{'flows/' ~ taskrun.value}}"
        includeChildNamespaces: false

      - id: scripts
        type: io.kestra.plugin.git.SyncNamespaceFiles
        namespace: "{{ taskrun.value }}"
        gitDirectory: "{{'scripts/' ~ taskrun.value}}"

pluginDefaults:
  - type: io.kestra.plugin.git
    values:
      username: anna-geller
      url: https://github.com/anna-geller/product
      password: "{{ secret('GITHUB_ACCESS_TOKEN') }}"
      branch: main
      dryRun: false

triggers:
  - id: every_full_hour
    type: io.kestra.plugin.core.trigger.Schedule
    cron: "0 * * * *"

Create a flow that clones a Git repository and runs a Python script
id: clone_and_run_python
namespace: company.team

tasks:
  - id: working_directory
    type: io.kestra.plugin.core.flow.WorkingDirectory
    tasks:
      - id: clone_repository
        type: io.kestra.plugin.git.Clone
        url: https://github.com/kestra-io/examples
        branch: main
      - id: run_python_script
        type: io.kestra.plugin.scripts.python.Commands
        beforeCommands:
          - pip install requests pandas > /dev/null
        commands:
          - python examples/scripts/etl_script.py

## Postgres
Export a Postgres table called mytable to a CSV file
id: postgres_export_mytable
namespace: company.team

tasks:
  - id: export_mytable_csv
    type: io.kestra.plugin.jdbc.postgresql.CopyOut
    url: jdbc:postgresql://your_postgres_host:5432/your_database
    username: "{{ secret(\"POSTGRES_USERNAME\") }}"
    password: "{{ secret(\"POSTGRES_PASSWORD\") }}"
    table: mytable
    format: CSV
    header: true
    delimiter: ","

Query a Postgres table called mytable
id: query-postgres-table
namespace: company.team

tasks:
  - id: query_mytable
    type: io.kestra.plugin.jdbc.postgresql.Query
    url: jdbc:postgresql://{{ secret('POSTGRES_HOST') }}:5432/{{ secret('POSTGRES_DATABASE') }}
    username: "{{ secret('POSTGRES_USERNAME') }}"
    password: "{{ secret('POSTGRES_PASSWORD') }}"
    sql: |
      SELECT * FROM mytable;
    fetchType: FETCH


## MongoDB

Find documents in a MongoDB collection called mycollection
id: find_mycollection_documents
namespace: company.team

tasks:
  - id: find_docs
    type: io.kestra.plugin.mongodb.Find
    connection:
      uri: mongodb://root:example@localhost:27017/?authSource=admin
    database: my_database
    collection: mycollection

Load documents into a MongoDB mycollection using a file from input mydata
id: load_documents_to_mongodb
namespace: company.team

inputs:
  - id: mydata
    type: FILE

tasks:
  - id: load_to_mycollection
    type: io.kestra.plugin.mongodb.Load
    connection:
      uri: "mongodb://<user>:<password>@<host>:<port>/<authSource>"
    database: "my_database"
    collection: "mycollection"
    from: "{{ inputs.mydata }}"


## Airbyte
Trigger an Airbyte connection sync and retry it up to 3 times
id: airbyte_sync_with_retry
namespace: company.team

tasks:
  - id: sync_connection
    type: io.kestra.plugin.airbyte.connections.Sync
    url: http://localhost:8080
    connectionId: e3b1ce92-547c-436f-b1e8-23b6936c12cd
    retry:
      type: constant
      maxAttempt: 3
      interval: PT1M

Sync an Airbyte cloud job for connection qwertzuiop123456789 and retry it up to 3 times every 2 minutes
id: airbyte_cloud_sync_with_retry
namespace: company.team

tasks:
  - id: sync_connection
    type: io.kestra.plugin.airbyte.cloud.jobs.Sync
    connectionId: qwertzuiop123456789
    token: "{{ secret('AIRBYTE_CLOUD_TOKEN') }}"
    retry:
      type: constant
      interval: 2m
      maxAttempt: 3


## Orchestrators
Run an Airflow DAG called mydag

id: trigger_airflow_dag
namespace: dev

tasks:
  - id: run_mydag
    type: io.kestra.plugin.airflow.dags.TriggerDagRun
    baseUrl: http://host.docker.internal:8080
    dagId: mydag
    wait: true


Orchestrate an Ansible playbook stored in Namespace Files

id: run_ansible_from_namespace
namespace: company.team

tasks:
  - id: run_playbook
    type: io.kestra.plugin.ansible.cli.AnsibleCLI
    namespaceFiles:
      enabled: true
    commands:
      - ansible-playbook -i inventory.ini myplaybook.yml


## DuckDB

Run a DuckDB query that reads a CSV file
id: duckdb_read_csv
namespace: dev

tasks:
  - id: download_csv
    type: io.kestra.plugin.core.http.Download
    uri: https://huggingface.co/datasets/kestra/datasets/raw/main/csv/orders.csv

  - id: query_csv
    type: io.kestra.plugin.jdbc.duckdb.Query
    url: "jdbc:duckdb:"
    sql: SELECT * FROM read_csv_auto('input.csv', header=True);
    inputFiles:
      input.csv: "{{ outputs.download_csv.uri }}"
    fetchType: FETCH


## AWS ECR
Fetch AWS ECR authorization token to push Docker images to Amazon ECR

id: fetch_ecr_auth_token
namespace: company.team

tasks:
  - id: get_ecr_token
    type: io.kestra.plugin.aws.ecr.GetAuthToken
    accessKeyId: "{{ secret('AWS_ACCESS_KEY_ID') }}"
    secretKeyId: "{{ secret('AWS_SECRET_ACCESS_KEY') }}"
    region: "{{ secret('AWS_REGION') }}"

## Kafka

Run a flow whenever 5 records are available in Kafka topic mytopic

id: kafka_trigger_5_records
namespace: dev

tasks:
  - id: log_messages
    type: io.kestra.plugin.core.log.Log
    message: "Flow triggered by Kafka messages. Consumed {{ trigger.messagesCount }} messages."

triggers:
  - id: kafka_message_count
    type: io.kestra.plugin.kafka.Trigger
    topic: mytopic
    properties:
      bootstrap.servers: localhost:9092 # Replace with your Kafka broker(s)
    groupId: kestra_mytopic_consumer
    maxRecords: 5
    keyDeserializer: STRING
    valueDeserializer: STRING

## Databricks

Submit a run for a Databricks job

id: databricks-submit-run-flow
namespace: dev.kestra

tasks:
  - id: submit-databricks-job-run
    type: io.kestra.plugin.databricks.job.SubmitRun
    host: "{{ secret('DATABRICKS_HOST') }}"
    authentication:
      token: "{{ secret('DATABRICKS_TOKEN') }}"
    runTasks:
      - existingClusterId: <your-cluster-id>
        taskKey: exampleTaskKey
        sparkPythonTask:
          pythonFile: /Shared/example.py
          sparkPythonTaskSource: WORKSPACE
    waitForCompletion: PT10M