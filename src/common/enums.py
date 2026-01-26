from enums import Enum


class AzureClientEnum(str, Enum):
    blob = "blob"
    event_grid = "event_grid"
    identity = "identity"
    event_grid = "event_grid"
    event_bus = "evnt_bus"
    sql = "sql"
