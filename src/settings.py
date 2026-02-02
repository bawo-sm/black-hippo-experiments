# endpoint urls
HOME_URL = "/"
SIM_SEARCH_CLASSIFICATION_URL = "/sim_search_classification"
GET_STATUS_URL = "/get_status"
GET_ITEMS_URL = "/get_items"
CHECK_ITEMS_URL = "/check_items"
CREATE_REFERENCE_DATA_URL = "/create_reference_data"
DELETE_REFERENCE_DATA_URL = "/delete_reference_data"


# vector db qdrant
SIM_SEARCH_INDEX_NAME = "reference_data"
RESULTS_INDEX_NAME = "results_data"
EMBEDDING_SIZE = 768


# storage account
IMAGES_CONTAINER = "images"


# other
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM = "gpt-4o-mini"