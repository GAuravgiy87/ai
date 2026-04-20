import sqlite3

def check_cameras():
    try:
        conn = sqlite3.connect('db.sqlite3')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM cameras')
        rows = cursor.fetchall()
        print(f"Found {len(rows)} cameras:")
        for row in rows:
            print(f"ID: {row['camera_id']}, Source: {row['source']}")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_cameras()
