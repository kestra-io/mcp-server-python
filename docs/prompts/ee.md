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

List all worker groups.
Create a worker group with key "group1" and description "Main group".
Get details for the worker group with ID "abc123".
Update the worker group with ID "abc123" to have key "group1" and description "Updated group".
Delete the worker group with ID "abc123".

Enter maintenance mode.
Exit maintenance mode.

List all announcements.
Create an announcement with message "System maintenance", type "WARNING", start date "2025-12-28T00:00:00Z", and end date "2025-12-30T00:00:00Z".
Update the announcement with ID "banner1" to message "Maintenance extended", type "WARNING", start date "2025-12-30T00:00:00Z", and end date "2025-12-31T00:00:00Z".
Delete the announcement with ID "banner1".

Search for users with the email "alice@example.com".
List all users in the tenant "main".
Search for users with superadmin permission.

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
