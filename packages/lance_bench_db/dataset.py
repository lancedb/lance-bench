"""Dataset connection utilities."""

import os
from pathlib import Path

import lancedb


def get_database_uri() -> str:
    """Get the database URI for connecting to LanceDB.

    This function checks for the LANCE_BENCH_URI environment variable first.
    If not set, it returns the path to .lance-bench in the user's home directory.

    Returns:
        str: The database URI

    Raises:
        ValueError: If the home directory cannot be determined
    """
    # Check for environment variable first
    uri = os.environ.get("LANCE_BENCH_URI")
    if uri:
        return uri

    # Fall back to home directory
    home_dir = Path.home()
    db_path = home_dir / ".lance-bench"

    return str(db_path)


def connect() -> lancedb.DBConnection:
    """Connect to the LanceDB database.

    This function uses get_database_uri() to determine the database location
    and returns a database connection.

    Returns:
        lancedb.DBConnection: The database connection

    Raises:
        Exception: If connection fails
    """
    uri = get_database_uri()

    # Connect to the database
    db = lancedb.connect(uri)
    return db
