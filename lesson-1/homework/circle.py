import math

class Circle:
    @staticmethod
    def calculate_area(radius):
        return math.pi * (radius ** 2)
    
    @staticmethod
    def calculate_circumference(radius):
        return 2 * math.pi * radius