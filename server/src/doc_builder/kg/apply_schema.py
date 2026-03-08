"""
Apply Neo4j schema (constraints, indexes, vector index).
"""

import asyncio
import logging
from pathlib import Path

from doc_builder.config import get_settings
from doc_builder.kg.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).parent / "schema.cypher"


async def apply_schema(create_vector_index: bool = True) -> dict:
    """
    Apply the Neo4j schema from schema.cypher file.
    
    Args:
        create_vector_index: Whether to create the vector index
        
    Returns:
        Summary of applied statements
    """
    client = get_neo4j_client()
    settings = get_settings()

    # Read schema file
    schema_content = SCHEMA_FILE.read_text()

    # Parse statements (skip comments and empty lines)
    statements = []
    current_statement = []

    for line in schema_content.split("\n"):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("//"):
            continue

        current_statement.append(line)

        # Statement ends with semicolon
        if line.endswith(";"):
            statement = " ".join(current_statement)
            # Remove trailing semicolon for Neo4j driver
            statement = statement.rstrip(";").strip()
            if statement:
                statements.append(statement)
            current_statement = []

    # Apply each statement
    results = {
        "total": len(statements),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    for statement in statements:
        try:
            await client.execute_write(statement)
            results["success"] += 1
            logger.debug(f"Applied: {statement[:50]}...")
        except Exception as e:
            # Many statements may already exist, which is fine
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent" in error_msg:
                results["success"] += 1
                logger.debug(f"Already exists: {statement[:50]}...")
            else:
                results["failed"] += 1
                results["errors"].append({"statement": statement[:100], "error": str(e)})
                logger.warning(f"Failed to apply: {statement[:50]}... Error: {e}")

    # Create vector index separately (requires special handling)
    if create_vector_index:
        try:
            vector_index_query = f"""
            CALL db.index.vector.createNodeIndex(
                '{settings.vector_index_name}',
                'DocChunk',
                'embedding',
                {settings.embedding_dimensions},
                'cosine'
            )
            """
            await client.execute_write(vector_index_query)
            results["vector_index"] = "created"
            logger.info(f"Created vector index: {settings.vector_index_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "equivalent" in error_msg:
                results["vector_index"] = "exists"
                logger.debug("Vector index already exists")
            else:
                results["vector_index"] = f"failed: {e}"
                logger.warning(f"Failed to create vector index: {e}")

    logger.info(
        f"Schema applied: {results['success']}/{results['total']} statements, "
        f"{results['failed']} failed"
    )

    return results


async def verify_schema() -> dict:
    """
    Verify that the schema is properly applied.
    
    Returns:
        Schema verification status
    """
    client = get_neo4j_client()

    # Check constraints
    constraints = await client.execute_query(
        "SHOW CONSTRAINTS WHERE name STARTS WITH 'doc_'"
    )

    # Check indexes
    indexes = await client.execute_query(
        "SHOW INDEXES WHERE name STARTS WITH 'doc_'"
    )

    # Check vector index
    vector_indexes = await client.execute_query(
        "SHOW INDEXES WHERE type = 'VECTOR'"
    )

    return {
        "constraints": len(constraints),
        "indexes": len(indexes),
        "vector_indexes": len(vector_indexes),
        "constraint_names": [c.get("name") for c in constraints],
        "index_names": [i.get("name") for i in indexes],
        "vector_index_names": [v.get("name") for v in vector_indexes],
    }


def main():
    """CLI entry point for schema application."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def run():
        try:
            print("Applying Neo4j schema...")
            result = await apply_schema()
            print(f"\nResults:")
            print(f"  Success: {result['success']}/{result['total']}")
            print(f"  Failed: {result['failed']}")
            print(f"  Vector Index: {result.get('vector_index', 'not attempted')}")

            if result["errors"]:
                print("\nErrors:")
                for err in result["errors"]:
                    print(f"  - {err['statement'][:60]}...")
                    print(f"    {err['error']}")

            print("\nVerifying schema...")
            verify = await verify_schema()
            print(f"  Constraints: {verify['constraints']}")
            print(f"  Indexes: {verify['indexes']}")
            print(f"  Vector Indexes: {verify['vector_indexes']}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        finally:
            from doc_builder.kg.neo4j import close_neo4j_client
            await close_neo4j_client()

    asyncio.run(run())


if __name__ == "__main__":
    main()
