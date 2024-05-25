# REMLA-TEAM-15
## Todo:
• Include the execution of the two linters in a Github workflow that can fail the build.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/REMLA24-TEAM-15/REMLA-GROUP-15.git

2. **Navigate to the project directory:**
   ```bash
   cd REMLA-GROUP-15

3. **Create a virtual environment:**
   ```bash
   python -m venv venv

4. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/MacOS:**
     ```bash
     source venv/bin/activate
     ```


6. **Install the dependencies from requirements.txt:**
   ```bash
   pip install -r requirements.txt

7. **DVC:**
   ```bash
   dvc pull
   dvc repro

8. **Running Pylint to check the code quality:**
   ```bash
   pylint src
  
9. **Running flake8 to check the code quality:**
   - **If it does not return anything, the code quality is good:**
     ```bash
     venv\Scripts\activate
     ```
