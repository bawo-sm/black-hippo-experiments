from enum import Enum


class AzureClientEnum(str, Enum):
    blob = "blob"
    event_grid = "event_grid"
    identity = "identity"
    event_bus = "evnt_bus"
    sql = "sql"


class TaskStatusEnum(str, Enum):
    in_progress = "in_progress"
    success = "success"
    error = "error"


class TaskEnum(str, Enum):
    create_reference_data = "create_reference_data"
    classification = "classification"
    color_recognition = "color_recognition"
    hs_code = "hs_code"
