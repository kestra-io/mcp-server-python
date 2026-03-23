Invite a user with email alice@example.com to the group "admins".
Invite a user with email bob@example.com without assigning any group.

Create a unit test from this YAML definition: <paste YAML here>.
Run the unit test with namespace "dev" and id "my-test".
Delete the unit test with namespace "dev" and id "my-test".

Create an app from this YAML definition: <paste YAML here>.
Enable the app with UID "my-app-uid".
Disable the app with UID "my-app-uid".
Delete the app with UID "my-app-uid".
Search for apps in namespace "dev".
Search for apps with the tag "production".

List all announcements.
Create an announcement with message "System maintenance", type "WARNING", start date "2025-12-28T00:00:00Z", and end date "2025-12-30T00:00:00Z".
Update the announcement with ID "banner1" to message "Maintenance extended", type "WARNING", start date "2025-12-30T00:00:00Z", and end date "2025-12-31T00:00:00Z".
Delete the announcement with ID "banner1".

---

Invite anna@kestra.id to tenant demo and group Admins

Create a unit test from this YAML:
id: test_microservices_workflow
flowId: microservices-and-apis
namespace: tutorial
testCases:
  - id: server_should_be_reachable
    type: io.kestra.core.tests.flow.UnitTest
    fixtures:
      inputs:
        server_uri: https://kestra.io
    assertions:
      - value: "{{outputs.http_request.code}}"
        equalTo: 200

  - id: server_should_be_unreachable
    type: io.kestra.core.tests.flow.UnitTest
    fixtures:
      inputs:
        server_uri: https://kestra.io/bad-url
      tasks:
        - id: server_unreachable_alert
          description: no Slack message from tests
    assertions:
      - value: "{{outputs.http_request.code}}"
        notEqualTo: 200

Update the unit test as follows:

id: test_microservices_workflow
flowId: microservices-and-apis
namespace: tutorial
testCases:
  - id: server_should_be_reachable
    type: io.kestra.core.tests.flow.UnitTest
    fixtures:
      inputs:
        server_uri: https://kestra.io
    assertions:
      - value: "{{outputs.http_request.code}}"
        equalTo: 200
        description: Reachable URL as expected

  - id: server_should_be_unreachable
    type: io.kestra.core.tests.flow.UnitTest
    fixtures:
      inputs:
        server_uri: https://kestra.io/bad-url
      tasks:
        - id: server_unreachable_alert
          description: no Slack message from tests
    assertions:
      - value: "{{outputs.http_request.code}}"
        notEqualTo: 200
        description: Unreachable URL as expected

Run the unit test test_microservices_workflow in the tutorial namespace

Generate a unit test for the flow hello-world in namespace tutorial
Generate a unit test for the flow hello-world in namespace tutorial and auto-create it

Generate an app that lets users trigger the hello-world flow with a form
Generate an app for monitoring executions in the tutorial namespace and auto-create it
