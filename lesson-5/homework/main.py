roster = [
    ["Benjamin Sisko", "Human", 40],
    ["Jadzia Dax", "Trill", 300],
    ["Kira Nerys", "Bajoran", 29]
]

for row in roster:
    if row[0] == "Jadzia Dax":
        row[0] = "Ezri Dax"

for row in roster:
    if row[1] == "Bajoran":
        print(f"Name: {row[0]}, Age: {row[2]}")
