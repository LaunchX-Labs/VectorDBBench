import logging
import json
import pickle
import time
import numpy as np
from contextlib import contextmanager
import mysql.connector
from mysql.connector import Error


from ..api import DBCaseConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import FacebookMyRocksConfig

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class FacebookMyRocks(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        print(db_config)

        self.user = db_config.get("user", "root")
        self.password = db_config.get("password")
        self.host = db_config.get("host", "localhost")
        self.database = db_config.get("database")
        self.vector_type = db_config.get("vector_type", "JSON")
        self.metric_type = db_config.get("metric_type", "COSINE")
        self.index_type = db_config.get("index_type", "ivfpq")
        self.table_name = "vec_collection"
        self.trained_index_table = "VECTORDB_DATA"

        if self.index_type.lower() == "ivfflat":
            self.trained_index_id = "cohere_wiki_ivfflat"
        else:
            self.trained_index_id = "cohere_wiki_ivfpq"

        self.dimension = db_config.get("dimension", 768)
        self.name = "FacebookMyRocks"

        # construct basic units
        self.conn, self.cursor = self._create_connection()
        log.debug("Creating connection in start __init__ method")

        # Drop table if exists
        drop_query = f"DROP TABLE IF EXISTS {self.trained_index_table}"
        self.cursor.execute(drop_query)
        
        # Drop user if exists
        self.cursor.execute("DROP USER IF EXISTS 'admin:sys.database'")

        if self.index_type in ['ivfflat', 'ivfpq']:
            # Create the VECTORDB_DATA table using the dedicated function
            self._create_vectordb_data_table()
             
            # Create user
            self.cursor.execute("CREATE USER 'admin:sys.database'")

            # Grant privileges
            self.cursor.execute("GRANT ALL ON *.* TO 'admin:sys.database'@'%'")

            # Flush privileges
            self.cursor.execute("FLUSH PRIVILEGES")

            if self.index_type == "ivfflat":
                if self.metric_type.lower() == "cosine":
                    self._execute_sql_file("cohere_wiki_ivfflat_cosine.sql")
                elif self.metric_type.lower() == "l2":
                    self._execute_sql_file("cohere_wiki_ivfflat_l2.sql")
                elif self.metric_type.lower() == "ip":
                    self._execute_sql_file("cohere_wiki_ivfflat_ip.sql")
                else:
                    log.error("Invalid metric type chosen")
            else:
                if self.metric_type.lower() == "cosine":
                    self._execute_sql_file("cohere_wiki_ivfpq_cosine.sql")
                elif self.metric_type.lower() == "l2":
                    self._execute_sql_file("cohere_wiki_ivfpq_l2.sql")
                elif self.metric_type.lower() == "ip":
                    self._execute_sql_file("cohere_wiki_ivfpq_ip.sql")
                else:
                    log.error("Invalid metric type chosen")

            # Commit all changes
            self.conn.commit()


        self._drop_table(self.table_name)
        self._create_table(self.dimension)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None



    def _create_connection(self):
        try:
            # Create connection
            conn = mysql.connector.connect(
                host=self.host,       # MySQL server address
                user=self.user,            # MySQL username
                password=self.password, # MySQL password
                database=self.database      # Database name
            )

            if conn.is_connected():
                log.debug("✅ Connected to MySQL database")

                # Create a cursor
                cursor = conn.cursor()

                assert cursor is not None, "Cursor is not initialized"

                return conn, cursor
            
        except Error as e:
            logging.error("❌ Error while connecting to MySQL: %s", e)

    
    def _create_table(self, dim):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self.vector_type == "BLOB":
            vector_column = f"v BLOB NOT NULL FB_VECTOR_DIMENSION {dim}"
        else:
            vector_column = f"v JSON NOT NULL FB_VECTOR_DIMENSION {dim}"

        if self.index_type == "flat":
            create_table_sql = f"""
                CREATE TABLE {self.table_name} (
                    id BIGINT NOT NULL PRIMARY KEY,
                    {vector_column},
                    name VARCHAR(64) COLLATE utf8mb4_bin,
                    label VARCHAR(64) COLLATE utf8mb4_bin, -- added label field
                    INDEX vector_idx(v) FB_VECTOR_INDEX_TYPE '{self.index_type}'
                ) ENGINE=rocksdb
            """

        elif self.index_type in ["ivfflat", "ivfpq"]:
            create_table_sql = f"""
                CREATE TABLE {self.table_name} (
                    id BIGINT NOT NULL PRIMARY KEY,
                    {vector_column},
                    name VARCHAR(64) COLLATE utf8mb4_bin,
                    label VARCHAR(64) COLLATE utf8mb4_bin, -- added label field
                    INDEX vector_idx(v) FB_VECTOR_INDEX_TYPE '{self.index_type}' FB_VECTOR_TRAINED_INDEX_TABLE '{self.trained_index_table}' FB_VECTOR_TRAINED_INDEX_ID '{self.trained_index_id}'
                ) ENGINE=rocksdb
            """

        self.cursor.execute(create_table_sql)
        self.conn.commit()


    def _train_vector_index(self):
        """Train vector index after data insertion (for IVF indexes)(analyzing table for flat index as well)"""

        log.info("Training vector index...")
        try:
            self.cursor.execute("SET GLOBAL rocksdb_table_stats_use_table_scan = ON")
            self.conn.commit()
            
            # ANALYZE TABLE returns rows -> must fetch them
            self.cursor.execute(f"ANALYZE TABLE {self.table_name}")
            while True:
                _ = self.cursor.fetchall()
                if not self.cursor.next_result():
                    break
            
            self.cursor.execute("SET GLOBAL rocksdb_table_stats_use_table_scan = DEFAULT")
            self.conn.commit()
            
            # Check if training was successful
            self.cursor.execute(f"""
                SELECT NTOTAL, AVG_LIST_SIZE 
                FROM INFORMATION_SCHEMA.ROCKSDB_VECTOR_INDEX 
                WHERE TABLE_NAME = '{self.table_name}'
            """)
            result = self.cursor.fetchone()
            
            if result and result[0] > 0:
                log.info(f"Index training successful: NTOTAL={result[0]}, AVG_LIST_SIZE={result[1]}")
            else:
                log.warning("Warning: Index training may have failed - NTOTAL is 0")
                    
        except Exception as e:
            log.error(f"Warning: Index training failed: {e}")



    def _create_vectordb_data_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        create_table_sql = f"""
            CREATE TABLE {self.trained_index_table} (
                id VARCHAR(128) NOT NULL,
                type VARCHAR(128) NOT NULL,
                seqno INT NOT NULL,
                value JSON NOT NULL,
                PRIMARY KEY (id, type, seqno)
            )
        """

        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def _execute_sql_file(self, sql_file_path):
        """
        Execute SQL commands from a file (alternative to MySQL 'source' command)
        Note: This reads the file and executes statements, but 'source' itself 
        cannot be run through cursor.execute()
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        
        try:
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                sql_content = file.read()
                
            # Remove comments and split by semicolon
            statements = []
            for line in sql_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('--') and not line.startswith('#'):
                    statements.append(line)
            
            full_sql = ' '.join(statements)
            statements = [stmt.strip() for stmt in full_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    self.cursor.execute(statement)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
        except Exception as e:
            raise Exception(f"Error executing SQL file {sql_file_path}: {str(e)}")

    def _drop_table(self, table_name):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table: {table_name}")

        # Flush tables before dropping to avoid locking issues
        self.cursor.execute("FLUSH TABLES")
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.commit()  # Better to use connection's commit method
        self.cursor.execute("FLUSH TABLES")

    def _vector_to_json(self, vector) -> str:
        """Convert vector to JSON string format (works with FB_VECTOR_IP)"""
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return json.dumps(vector)

    @classmethod
    def config_cls(cls) -> type[FacebookMyRocksConfig]:
        return FacebookMyRocksConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig


    @contextmanager
    def init(self):
        """create and destoy connections to database."""
        self.conn, self.cursor = self._create_connection()

        if self.metric_type == "COSINE":
            if self.vector_type == "BLOB":
                # Pre-normalize during insertion for optimal cosine similarity performance
                self.insert_sql = f"""
                    INSERT INTO {self.table_name} (id, v, name, label) 
                    VALUES (%s, FB_VECTOR_JSON_TO_BLOB(FB_VECTOR_NORMALIZE_L2(%s)), %s, %s)
                """
            else:  # JSON
                # Pre-normalize JSON vectors during insertion
                self.insert_sql = f"""
                    INSERT INTO {self.table_name} (id, v, name, label) 
                    VALUES (%s, FB_VECTOR_NORMALIZE_L2(%s), %s, %s)
                """
        else:
            if self.vector_type == "BLOB":
                # Pre-normalize during insertion for optimal cosine similarity performance
                self.insert_sql = f"""
                    INSERT INTO {self.table_name} (id, v, name, label) 
                    VALUES (%s, FB_VECTOR_JSON_TO_BLOB(%s), %s, %s)
                """
            else:  # JSON
                # Pre-normalize JSON vectors during insertion
                self.insert_sql = f"""
                    INSERT INTO {self.table_name} (id, v, name, label) 
                    VALUES (%s, %s, %s, %s)
                """

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            log.debug("Closed DB connection in init context manager")
            # time.sleep(5)
            self.cursor = None
            self.conn = None


    def optimize(self, data_size: int | None = None):
        pass


    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """

        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        assert len(embeddings) == len(metadata)

        try:
            batch_data = []
            for i in range(len(embeddings)):
                vector_json = self._vector_to_json(embeddings[i])
                name = f"cohere_vector_{metadata[i]}"
                label = labels_data[i] if labels_data else None
                batch_data.append((int(metadata[i]), vector_json, name, label))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.conn.commit()

            # Check if we have exactly 1M vectors before training index
            self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            vector_count = self.cursor.fetchone()[0]

            if vector_count == 1000000:
                log.info(f"Found exactly 1,000,000 vectors. Training vector index...")
                # Train index after all data is inserted
                self._train_vector_index()
            else:
                log.debug(f"Vector count is {vector_count}, not 1,000,000. Skipping index training.")

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            return 0, e
        

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        
        if self.conn==None or self.cursor==None:
            self._create_connection()

        # Set nprobe to 64 after creating connection
        if self.cursor is not None:
            try:
                self.cursor.execute("SET SESSION fb_vector_search_nprobe = 32")
                log.info("✅ Set fb_vector_search_nprobe to 32")
            except Exception as e:
                log.warning(f"⚠️ Warning: Failed to set nprobe: {e}")

        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        query_json = self._vector_to_json(query)
        
        try:
            if self.metric_type == "COSINE":
                # Step 1: Normalize the query vector
                normalize_sql = "SELECT FB_VECTOR_NORMALIZE_L2(%s) as normalized_query"
                
                log.debug(f"Step 1 - Normalizing query: {normalize_sql}")
                
                self.cursor.execute(normalize_sql, [query_json])
                result = self.cursor.fetchone()
                
                if not result:
                    log.error("Failed to normalize query vector")
                    return []
                
                normalized_query = result[0]
                log.debug(f"Step 1 complete - Normalized query")
                
                # Step 2: Execute main search with pre-normalized query
                # This should use the index because 'v' is the raw column
                main_sql = f"""
                    SELECT id, name, FB_VECTOR_IP(v, %s) AS dis
                    FROM {self.table_name}
                    ORDER BY dis DESC
                    LIMIT %s
                """
                
                params = [normalized_query, k]
                
                log.debug(f"Step 2 - Executing main search: {main_sql}")
                log.debug(f"With normalized query and k={k}")
                
                self.cursor.execute(main_sql, params)
                rows = self.cursor.fetchall()
                
                log.debug(f"Search completed - Found {len(rows)} results")
                
                # Extract IDs
                ids = [row[0] for row in rows]
                return ids
            
            elif self.metric_type == "IP":
                main_sql = f"""
                    SELECT id, name, FB_VECTOR_IP(v, %s) AS dis
                    FROM {self.table_name}
                    ORDER BY dis DESC
                    LIMIT %s
                """
                
                params = [query_json, k]
                
                log.debug(f"Executing main search: {main_sql}")
                log.debug(f"With k={k}")
                
                self.cursor.execute(main_sql, params)
                rows = self.cursor.fetchall()
                
                log.debug(f"Search completed - Found {len(rows)} results")
                
                # Extract IDs
                ids = [row[0] for row in rows]
                return ids
            
            elif self.metric_type == "L2":
                main_sql = f"""
                    SELECT id, name, FB_VECTOR_L2(v, %s) AS dis
                    FROM {self.table_name}
                    ORDER BY dis ASC
                    LIMIT %s
                """
                
                params = [query_json, k]
                
                log.debug(f"Executing main search: {main_sql}")
                log.debug(f"With k={k}")
                
                self.cursor.execute(main_sql, params)
                rows = self.cursor.fetchall()
                
                log.debug(f"Search completed - Found {len(rows)} results")
                
                # Extract IDs
                ids = [row[0] for row in rows]
                return ids
            
            else:
                log.error("Invalid metric type")
                return []


        except Exception as e:
            log.warning(f"Failed to search embeddings: {e}", exc_info=True)
            return []