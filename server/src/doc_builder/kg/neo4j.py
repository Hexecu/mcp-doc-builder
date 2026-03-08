"""
Neo4j async client with connection pooling and singleton pattern.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from doc_builder.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Async Neo4j client with connection pooling.
    
    Uses singleton pattern to ensure single driver instance.
    """

    _instance: "Neo4jClient | None" = None
    _driver: AsyncDriver | None = None
    _lock: asyncio.Lock | None = None

    def __new__(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lock = asyncio.Lock()
        return cls._instance

    async def connect(self, settings: Settings | None = None) -> None:
        """
        Initialize the Neo4j driver connection.
        
        Args:
            settings: Optional settings override
        """
        if self._driver is not None:
            return

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._driver is not None:
                return

            settings = settings or get_settings()

            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}")

            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
            )

            # Verify connectivity
            try:
                await self._driver.verify_connectivity()
                logger.info("Neo4j connection established successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                await self.close()
                raise

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    @asynccontextmanager
    async def session(self, database: str = "neo4j"):
        """
        Get an async session context manager.
        
        Args:
            database: Database name (default: neo4j)
            
        Yields:
            AsyncSession instance
        """
        if self._driver is None:
            await self.connect()

        session = self._driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """
        Execute a read query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            List of result records as dictionaries
        """
        if self._driver is None:
            await self.connect()

        try:
            result = await self._driver.execute_query(
                query,
                parameters_=parameters or {},
                database_=database,
            )
            return [dict(record) for record in result.records]
        except (ServiceUnavailable, SessionExpired) as e:
            logger.warning(f"Neo4j connection issue, retrying: {e}")
            await self.close()
            await self.connect()
            result = await self._driver.execute_query(
                query,
                parameters_=parameters or {},
                database_=database,
            )
            return [dict(record) for record in result.records]

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> dict[str, Any]:
        """
        Execute a write query and return summary.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            Query execution summary
        """
        if self._driver is None:
            await self.connect()

        async with self.session(database=database) as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }

    async def execute_write_return(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """
        Execute a write query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            List of result records as dictionaries
        """
        if self._driver is None:
            await self.connect()

        async with self.session(database=database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def health_check(self) -> dict[str, Any]:
        """
        Check Neo4j connection health.
        
        Returns:
            Health status dictionary
        """
        try:
            if self._driver is None:
                await self.connect()

            result = await self.execute_query("RETURN 1 as ping")
            return {
                "status": "healthy",
                "connected": True,
                "ping": result[0]["ping"] if result else None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }


# Global client instance
_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """Get the global Neo4j client instance."""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


async def close_neo4j_client() -> None:
    """Close the global Neo4j client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
