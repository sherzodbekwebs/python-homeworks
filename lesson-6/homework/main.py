import sqlite3

# Connect to (or create) the database file
conn = sqlite3.connect("mydatabase.db")
cursor = conn.cursor()

# Create the Roster table with the required fields
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Roster (
        Name TEXT,
        Species TEXT,
        Age INTEGER
    )
""")

# Insert the provided data into the Roster table
data = [
    ('Benjamin Sisko', 'Human', 40),
    ('Jadzia Dax', 'Trill', 300),
    ('Kira Nerys', 'Bajoran', 29)
]
cursor.executemany("INSERT INTO Roster (Name, Species, Age) VALUES (?, ?, ?)", data)
conn.commit()

# Update the Name of 'Jadzia Dax' to 'Ezri Dax'
cursor.execute("UPDATE Roster SET Name = ? WHERE Name = ?", ('Ezri Dax', 'Jadzia Dax'))
conn.commit()

# Query the database: Display the Name and Age of everyone classified as 'Bajoran'
cursor.execute("SELECT Name, Age FROM Roster WHERE Species = ?", ('Bajoran',))
results = cursor.fetchall()

# Print the query results
for name, age in results:
    print(f"Name: {name}, Age: {age}")

# Close the connection
conn.close()
