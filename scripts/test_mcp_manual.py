import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

# Load .env from the project root even when this script is run from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# psycopg async needs SelectorEventLoop on Windows (same as app/main.py).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.db.session import SessionLocal
from app.mcp import MCPError, MCPToolContext, build_postgres_mcp
from app.mcp.errors import error_response_dict

USER_ID = UUID("bfe941a1-8af4-47d9-a26a-4990e0def527")  # subash.s@fynmobility.com (OWNER)
WORKSPACE_ID = UUID("8b87a6d5-5f8d-4bd1-8d70-6591ed906e0b")
DS = UUID("b07e5c49-99dd-44b8-918f-ebba690da09f")  # test_infynity
TABLE_ID = UUID("a664475a-bf66-48d0-842f-879948b8da42")  # cities

session = SessionLocal()
registry, client = build_postgres_mcp(session)
ctx = MCPToolContext(workspace_id=WORKSPACE_ID, user_id=USER_ID)


async def call(name, args):
    try:
        r = await client.call_tool(name, args, ctx)
        print(f"\nOK {name}")
        print(r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
        return r
    except MCPError as e:
        print(f"\nERR {name}")
        print(error_response_dict(e))


async def main():
    print("Tools:", [t.name for t in registry.list_tools()])

    await call("postgres.list_schemas", {"data_source_id": DS})
    await call("postgres.list_tables", {"data_source_id": DS})
    await call("postgres.describe_table", {"data_source_id": DS, "table_id": TABLE_ID})
    await call("postgres.get_columns", {"data_source_id": DS, "table_id": TABLE_ID})
    await call("postgres.get_relationships", {"data_source_id": DS})

    await call(
        "postgres.sample_rows",
        {"data_source_id": DS, "table_id": TABLE_ID, "limit": 5},
    )

    await call(
        "postgres.query",
        {"data_source_id": DS, "sql": "SELECT 1 AS n", "limit": 10},
    )
    await call(
        "postgres.query",
        {
            "data_source_id": DS,
            "sql": "SELECT id, name FROM public.cities LIMIT 5",
            "limit": 5,
        },
    )

    await call("postgres.query", {"data_source_id": DS, "sql": "DELETE FROM cities"})


asyncio.run(main())
session.close()
