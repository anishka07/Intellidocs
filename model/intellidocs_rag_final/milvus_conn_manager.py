import logging
from pymilvus import connections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusConnectionManager:
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.connection_params = {
            "host": self.host,
            "port": self.port
        }

    def _connect(self, alias: str) -> None:
        """Establish connection with proper error handling"""
        try:
            if alias not in connections.list_connections():
                connections.connect(alias=alias, **self.connection_params)
                logger.info(f"Connected to Milvus with alias: {alias}")
            else:
                logger.info(f"Using existing connection for alias: {alias}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus with alias {alias}: {str(e)}")
            raise

    def ensure_connections(self) -> None:
        """Ensure both default and embedding connections exist"""
        self._connect("default")
        self._connect("em")

    def disconnect_all(self) -> None:
        """Safely disconnect all connections"""
        for alias in connections.list_connections():
            try:
                connections.disconnect(alias)
                logger.info(f"Disconnected from Milvus alias: {alias}")
            except Exception as e:
                logger.warning(f"Error disconnecting {alias}: {str(e)}")
