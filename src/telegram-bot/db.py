import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()
USER_HISTORY_DB = os.getenv("USER_HISTORY_DB")


def init():
    conn = sqlite3.connect(USER_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_tags (
            user_id INTEGER PRIMARY KEY,
            tags TEXT
        )
    ''')
    cursor.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message_text TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
    conn.commit()
    conn.close()


def get_user_tags(user_id):
    conn = sqlite3.connect(USER_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT tags FROM user_tags WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return set(result[0].split(',')) if result else set()


def insert_message(user_id, message_text):
    conn = sqlite3.connect(USER_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO messages (user_id, message_text) VALUES (?, ?)
    ''', (user_id, message_text))
    conn.commit()
    conn.close()


def save_user_tags(user_id, tags):
    conn = sqlite3.connect(USER_HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_tags (user_id, tags) VALUES (?, ?)
    ''', (user_id, ','.join(tags)))
    conn.commit()
    conn.close()
