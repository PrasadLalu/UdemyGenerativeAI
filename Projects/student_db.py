import sqlite3 as db

# Connect to Sqlite
connection = db.connect("student.db")

# Create a cursor object to insert record, create table
cursor = connection.cursor()

# Create STUDENT table
table_info = """
    CREATE TABLE student(name VARCHAR(25), class VARCHAR(25), section VARCHAR(25), mark INT)
"""  

cursor.execute(table_info)

cursor.execute('''Insert Into STUDENT values('Peter','Data Science','A',90)''')
cursor.execute('''Insert Into STUDENT values('John','Data Science','B',100)''')
cursor.execute('''Insert Into STUDENT values('Jessica','Data Science','A',86)''')
cursor.execute('''Insert Into STUDENT values('Jacob','DEVOPS','A',50)''')
cursor.execute('''Insert Into STUDENT values('Nick','DEVOPS','A',35)''')


# Display all the records
print("The inserted record are: ")
data = cursor.execute('''SELECT * FROM student''')
for row in data:
    print(row)
    
    
# Commit changes in the database
connection.commit()
connection.close()
