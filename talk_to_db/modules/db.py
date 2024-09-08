import json
import psycopg2
from psycopg2.sql import SQL, Identifier
from datetime import datetime

class PostgresManager:
    """A context manager for managing PostgreSQL database connections.

    This class provides a convenient way to establish a connection to a PostgreSQL database,
    execute SQL queries, retrieve table definitions, and automatically close the connection
    when the context is exited.

    Attributes:
        conn: A psycopg2 connection object.
        cur: A psycopg2 cursor object.
    """
    
    def __init__(self):
        """Initializes the DatabaseConnection object."""

        self.conn = None
        self.cur = None

    def __enter__(self):
        """Enters the context manager, returning the PostgresManager object.

        Returns:
            The PostgresManager object.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the context manager, closing the database connection if necessary.

        Args:
            exc_type: The type of exception that occurred, if any.
            exc_val: The exception object, if any.
            exc_tb: The traceback object, if any.
        """
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def connect_with_url(self, url):
        """Connects to a PostgreSQL database using the specified URL.

        Args:
            url: The URL of the PostgreSQL database to connect to.
        """

        self.conn = psycopg2.connect(url)
        self.cur = self.conn.cursor()

    def run_sql(self, sql) -> str:
        """Executes a SQL query and returns the results as a JSON string.

        Args:
            sql: The SQL query to execute.

        Returns:
            JSON: result of the query.
        """

        try:
            self.cur.execute(sql)
            columns = [desc[0] for desc in self.cur.description]
            res = self.cur.fetchall()

            list_of_dicts = [(dict(zip(columns, row))) for row in res]

            json_result = json.dumps(list_of_dicts, indent=4, default=self.datetime_handler)

            # dumping the results to a file
            with open("results.json", "w") as f:
                f.write(json_result)

            return json_result
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            raise

    def datetime_handler(self, obj):
        """Handles datetime objects when serializing to JSON.

        Args:
            obj: The datetime object

        Returns:
            str: ISO format date as a string
        """

        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def get_table_definitions(self, table_name):
        """Retrieves the CREATE TABLE statement for a given table in the 'public' schema.

        Args:
            self: An instance of the class containing this method.
            table_name: The name of the table to retrieve the definition for.

        Returns:
            A string representing the CREATE TABLE statement for the specified table.
        """

        get_def_stmt = """
        SELECT pg_class.relname as tablename,
            pg_attribute.attnum,
            pg_attribute.attname,
            format_type(atttypid, atttypmod)
        FROM pg_class
        JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
        JOIN pg_attribute ON pg_attribute.attrelid = pg_class.oid
        WHERE pg_attribute.attnum > 0
            AND pg_attribute.attisdropped = false
            AND pg_class.relname = %s
            AND pg_namespace.nspname = 'public' -- Assuming we are only interested in the public schema
        """
        self.cur.execute(get_def_stmt, (table_name,))
        rows = self.cur.fetchall()
        create_table_stmt = "CREATE TABLE {} (\n".format(table_name)
        for row in rows:
            create_table_stmt += "{} {},\n".format(row[2], row[3])
        create_table_stmt = create_table_stmt.rstrip(",\n") + "\n);"
        return create_table_stmt

    def get_all_table_names(self):
        """Retrieves a list of all table names in the 'public' schema.

        Args:
            self: An instance of the class containing this method.

        Returns:
            A list of table names.
        """

        get_all_tables_stmt = (
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
        )
        try:
            self.cur.execute(get_all_tables_stmt)
        except Exception as e:
            print(f"Error retrieving table names: {e}")
            raise
        return [row[0] for row in self.cur.fetchall()]

    def get_table_definitions_for_prompt(self):
        """Retrieves the definitions of all tables in the 'public' schema as a formatted string.

        Args:
            self: An instance of the class containing this method.

        Returns:
            A string containing the definitions of all tables in the 'public' schema, separated by newline characters.
        """
        
        table_names = self.get_all_table_names()
        definitions = []
        for table_name in table_names:
            definitions.append(self.get_table_definitions(table_name))
        return "\n\n".join(definitions)

    def get_table_definition_map_for_embeddings(self):
        """Retrieves a mapping of table names to their definitions for use in embeddings.

        Returns:
            dict: A dictionary where keys are table names and values are their corresponding definitions.
        """
        table_names = self.get_all_table_names()
        definitions = {}
        for table_name in table_names:
            definitions[table_name] = self.get_table_definitions(table_name)
        return definitions

    def get_related_tables(self, table_list, n=2):
        """Retrieves a list of tables related to the given tables through foreign key references.

        This method queries the database to find tables that have foreign keys referencing the given tables,
        as well as tables that are referenced by the given tables. The results are combined and duplicates are removed.

        Args:
            table_list (list): A list of table names to find related tables for.
            n (int, optional): The maximum number of related tables to retrieve for each table. Defaults to 2.

        Returns:
            list: A list of unique table names related to the given tables through foreign key references.
        """

        related_tables_dict = {}

        for table in table_list:
            # Query to fetch tables that have foreign key referencing the given table
            self.cur.execute(
                """
                SELECT
                    a.relname AS table_name
                FROM
                    pg_constraint con
                    JOIN pg_class a ON a.oid = con.conrelid
                WHERE
                    confrelid = (SELECT oid FROM pg_class WHERE relname = %s)
                LIMIT %s;
                """,
                (table, n),
            )

            related_tables = [row[0] for row in self.cur.fetchall()]

            # Query to fetch tables that the given table references
            self.cur.execute(
                """
                SELECT
                    a.relname AS referenced_table_name
                FROM
                    pg_constraint con
                    JOIN pg_class a ON a.oid = con.confrelid
                WHERE
                    conrelid = (SELECT oid FROM pg_class WHERE relname = %s)
                LIMIT %s;
                """,
                (table, n),
            )

            related_tables += [row[0] for row in self.cur.fetchall()]

            related_tables_dict[table] = related_tables

        # convert the dict to list and remove dups
        related_tables_list = []
        for table, related_tables in related_tables_dict.items():
            related_tables_list += related_tables

        related_tables_list = list(set(related_tables_list))

        return related_tables_list