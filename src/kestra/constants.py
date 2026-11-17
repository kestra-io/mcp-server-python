# Allowed execution states
_VALID_STATES = {
    "CREATED",
    "RUNNING",
    "PAUSED",
    "RESTARTED",
    "KILLING",
    "SUCCESS",
    "WARNING",
    "FAILED",
    "KILLED",
    "CANCELLED",
    "QUEUED",
    "RETRYING",
    "RETRIED",
    "SKIPPED",
}
# Every execution state that can be searched for: Kestra's `State.Type` enum.
# Wider than `_VALID_STATES`, which also gates `change_status`. SUBMITTED and
# RESUBMITTED exist from Kestra 1.1 on; a 1.0 server answers them with a 422.
_SEARCHABLE_STATES = _VALID_STATES | {"SUBMITTED", "BREAKPOINT", "RESUBMITTED"}
_VALID_FORCE_STATES = {"CREATED", "PAUSED", "QUEUED"}
_TERMINAL_STATES = {"SUCCESS", "WARNING", "FAILED", "KILLED", "CANCELLED", "SKIPPED"}

# Reserved keywords that collide with API routes
_RESERVED_FLOW_IDS = {
    "pause",
    "resume",
    "force-run",
    "change-status",
    "kill",
    "executions",
    "search",
    "source",
    "disable",
    "enable",
}
