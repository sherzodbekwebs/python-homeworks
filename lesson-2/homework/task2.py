import threading

def count_words(lines, result_list, index):
    """
    Berilgan qatorlar (lines) ichida so'zlarning uchrash sonini hisoblaydi va 
    natijani result_list[index] ga saqlaydi.
    """ 
    counts = {}
    for line in lines:
        words = line.split()
        for word in words:
            word = word.lower().strip('.,!?";:-()[]{}')
            if word:
                counts[word] = counts.get(word, 0) + 1
    result_list[index] = counts

def merge_counts(dict_list):
    final_counts = {}
    for d in dict_list:
        for word, count in d.items():
            final_counts[word] = final_counts.get(word, 0) + count
    return final_counts

def main():
    filename = "large_text.txt"  
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Fayl topilmadi: {filename}")
        return

    num_threads = 4
    total_lines = len(lines)
    chunk_size = total_lines // num_threads

    threads = []
    results = [None] * num_threads

    for i in range(num_threads):
        start = i * chunk_size
        if i == num_threads - 1:
            chunk = lines[start:]
        else:
            chunk = lines[start:start + chunk_size]
        
        t = threading.Thread(target=count_words, args=(chunk, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    final_counts = merge_counts(results)

    print("So'zlar uchrash sanalari:")
    for word, count in sorted(final_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()
