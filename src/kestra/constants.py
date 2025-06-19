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
