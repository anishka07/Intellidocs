from pymilvus import connections, utility
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_milvus():
    """Clean up all Milvus connections and collections"""
    try:
        # First, try to connect to Milvus if not connected
        if not connections.has_connection("default"):
            connections.connect(host="localhost", port="19530")

        # Get list of all collections
        collections = utility.list_collections()

        # Drop each collection
        for collection in collections:
            try:
                utility.drop_collection(collection)
                logger.info(f"Dropped collection: {collection}")
            except Exception as e:
                logger.error(f"Error dropping collection {collection}: {str(e)}")

        # Get list of all connection aliases
        existing_connections = connections.list_connections()

        # Disconnect all connections by their alias names
        for conn in existing_connections:
            if isinstance(conn, tuple):
                alias = conn[0]  # Extract the alias from the tuple
            else:
                alias = conn

            try:
                if connections.has_connection(alias):
                    connections.disconnect(alias)
                    logger.info(f"Disconnected from alias: {alias}")
            except Exception as e:
                logger.error(f"Error disconnecting {alias}: {str(e)}")

        logger.info("Milvus cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


def list_available_connections():
    """List available connections"""
    try:
        if not connections.has_connection("default"):
            connections.connect(host="localhost", port="19530")
        collections = utility.list_collections()
        for conn in collections:
            if isinstance(conn, tuple):
                alias = conn[0]  # Extract the alias from the tuple
            else:
                alias = conn
        return collections, alias
    except Exception as e:
        logger.error(f"Error during list_available_connections: {str(e)}")


if __name__ == "__main__":
    c, a = list_available_connections()
    print(c)
    print(a)
