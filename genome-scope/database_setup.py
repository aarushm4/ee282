import sqlite3

def setup_database():
    conn = sqlite3.connect('variants.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS variant_annotations (
            chrom TEXT,
            pos INTEGER,
            ref TEXT,
            alt TEXT,
            clin_sig TEXT
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    setup_database()
