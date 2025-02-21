import threading
import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def find_primes_in_range(start, end, result_list):
    local_primes = []  
    for number in range(start, end + 1):
        if is_prime(number):
            local_primes.append(number)
    result_list.extend(local_primes)

def main():
    start_range = 1
    end_range = 1000
    num_threads = 4

    range_size = (end_range - start_range + 1) // num_threads

    threads = []
    results = []  

    for i in range(num_threads):
        seg_start = start_range + i * range_size
        if i == num_threads - 1:
            seg_end = end_range
        else:
            seg_end = seg_start + range_size - 1

        t = threading.Thread(target=find_primes_in_range, args=(seg_start, seg_end, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    results.sort()  
    print("Topilgan tub sonlar:")
    print(results)

if __name__ == "__main__":
    main()







