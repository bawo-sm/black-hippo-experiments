from datetime import datetime, UTC
from sqlalchemy import create_engine, update, select, text
from sqlalchemy.orm import sessionmaker
from src.common.utils import get_env_variable
from src.common.db_schema import SQLItem, SQLTaskStatus, TaskStatusEnum, Base



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
    def get_engine():
        connection_string = (
            "mssql+pyodbc://{username}:{password}@{server}:{port}/{database}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
            "&Encrypt=yes"
            "&TrustServerCertificate=no"
            "&Connection Timeout=30"
        ).format(
            username=get_env_variable("SQL_USERNAME"),
            password=get_env_variable("SQL_PASSWORD"),
            server=get_env_variable("SQL_ENDPOINT"),
            port=get_env_variable("SQL_PORT"),
            database=get_env_variable("SQL_DATABASE"),
        )
        return create_engine(connection_string, echo=False)

    @staticmethod
    def get_session():
        return sessionmaker(bind=SQLService.get_engine())()

    @staticmethod
    def create_tables():
        Base.metadata.create_all(SQLService.get_engine())
    
    @staticmethod
    def drop_table(TableSchema):
        TableSchema.__table__.drop(SQLService.get_engine())

    @staticmethod
    def list_tables():
        rows = []
        with SQLService.get_engine().connect() as conn:
            result = conn.execute(text("SELECT * FROM information_schema.tables;"))
            for row in result:
                rows.append(row)
        return rows

    @staticmethod
    def trigger_insert(model):
        model.created_at = datetime.now(UTC)
        model.updated_at = datetime.now(UTC)

    @staticmethod
    def trigger_update(model):
        model.updated_at = datetime.now(UTC)
    