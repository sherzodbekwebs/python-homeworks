import os
import math

def create_virtual_env():
    os.system("python -m venv myenv")
    print("Virtual environment 'myenv' created.")

def install_packages():
    os.system("myenv\\Scripts\\pip install numpy pandas")  # Windows uchun
    print("Packages installed: numpy, pandas")