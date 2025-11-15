import mysql.connector
from mysql.connector import Error


try:
    # Create connection
    conn = mysql.connector.connect(
        host="localhost",       # MySQL server address
        user="root",            # MySQL username
        password="launchx", # MySQL password
        database="test_vectordb"      # Database name
    )

    if conn.is_connected():
        print("✅ Connected to MySQL database")

        # Create a cursor
        cursor = conn.cursor()

        # Example query
        cursor.execute("SELECT VERSION();")

        # Fetch result
        version = cursor.fetchone()
        print("MySQL version:", version[0])


except Error as e:
    print("❌ Error while connecting to MySQL:", e)

finally:
    # Close connection
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("🔒 MySQL connection closed")