from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ClientAuthenticationError
from logging import getLogger
from src.common.utils import get_env_variable
from src.settings import IMAGES_CONTAINER


logger = getLogger("BlobService")


class BlobService():
    
    def __init__(self):
        self.__blob_service_client = BlobServiceClient(
            account_url=f"https://{get_env_variable('ACCOUNT_NAME')}.blob.core.windows.net",
            credential=get_env_variable('STORAGE_ACCOUNT_KEY')
        )
        try:
            self.__blob_service_client.get_account_information()
            logger.info("Successfully connected.")
        except ClientAuthenticationError as e:
            logger.error("Connection refused")
            raise e
    
    def get_containers_names(self) -> list[str]:
        return [x["name"] for x in self.__blob_service_client.list_containers()]
    
    def get_image_url(self, image_name: str):
        self.check_file_exists(IMAGES_CONTAINER, image_name)
        return (
            f"https://{get_env_variable('ACCOUNT_NAME')}.blob.core.windows.net"
            f"/{IMAGES_CONTAINER}/{image_name}?{get_env_variable('STORAGE_ACCOUNT_KEY')}"
        )
    
    def check_file_exists(self, container_name: str, file_name: str):
        container_client = self.__blob_service_client.get_container_client(container_name)

        exists = False
        for blob in container_client.list_blobs():
            if blob.name == file_name:
                exists = True
                break
        
        assert exists

    def number_of_blobs(self, container_name: str):
        container_client = self.__blob_service_client.get_container_client(container_name)
        return len([x for x in container_client.list_blobs()])

    def get_blobs_names(self, container_name: str):
        container_client = self.__blob_service_client.get_container_client(container_name)
        return [x["name"] for x in container_client.list_blobs()]

    def upload_file(
            self, 
            local_filepath: str, 
            blob_name: str, 
            content_type: str,
            container_name: str
    ):
        blob_client = self.__blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )

        with open(local_filepath, "rb") as image:
            blob_client.upload_blob(
                image,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )