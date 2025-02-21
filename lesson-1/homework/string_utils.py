class StringUtils:
    @staticmethod
    def reverse_string(s):
        return s[::-1]
    
    @staticmethod
    def count_vowels(s):
        vowels = "aeiouAEIOU"
        return sum(1 for char in s if char in vowels)