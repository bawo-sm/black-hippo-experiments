from typing import Generator
from datetime import datetime, UTC
from sqlalchemy import create_engine, update, select
from sqlalchemy.orm import Session, sessionmaker
from src.common.utils import get_env_variable
from src.common.db_schema import SQLItem, SQLTaskStatus, TaskEnum, TaskStatusEnum



class SQLService:

    # crud like methods
    @staticmethod
    def insert(items: list[SQLTaskStatus] | list[SQLItem]):
        session = SQLService.get_session()
        session.add_all(items)
        session.commit()

    @staticmethod
    def load_items_by_origin_id(ids: list[int]) -> list[SQLItem]:
        return (
            SQLService.get_session()
            .query(SQLItem)
            .filter(SQLItem.origin_id.in_(ids))
            .all()
        )
    
    @staticmethod
    def set_task_status(item: SQLTaskStatus):
        session = SQLService.get_session()
        session.add(item)
        session.commit()

    @staticmethod
    def update_task_status(status: TaskStatusEnum, info: str, task_uuid: str):
        session = SQLService.get_session()
        stmt = (
            update(SQLTaskStatus)
            .where(SQLTaskStatus.task_uuid == task_uuid)
            .values(status=status, info=info)
        )
        session.execute(stmt)
        session.commit()

    @staticmethod
    def load_task_statuses(task_uuid: str | None = None) -> list[SQLTaskStatus]:
        session = SQLService.get_session()
        if task_uuid:
            stmt = select(SQLTaskStatus).where(SQLTaskStatus.task_uuid == task_uuid)
            results = session.execute(stmt).all()
        else:
            results = session.execute(select(SQLTaskStatus)).all()
        return results

    # database engine methods
    @staticmethod
    def get_database_url() -> str:
        db_user = get_env_variable("DB_USERNAME")
        db_password = get_env_variable("DB_PASSWORD")
        db_host = get_env_variable("DB_HOST_WRITE")
        db_port = get_env_variable("DB_PORT")
        db_name = get_env_variable("DB_DATABASE")

        uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return uri
    
    @staticmethod
    def get_engine():
        return create_engine(SQLService.get_database_url(), echo=False)

    @staticmethod
    def get_session():
        return sessionmaker(bind=SQLService.get_engine())()

    @staticmethod
    def trigger_insert(model):
        model.created_at = datetime.now(UTC)
        model.updated_at = datetime.now(UTC)

    @staticmethod
    def trigger_update(model):
        model.updated_at = datetime.now(UTC)
    