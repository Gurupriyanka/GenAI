import sqlite3

# Function to create tables
def create_tables():
    #initialize db
    conn = sqlite3.connect('employee.db')
    cursor = conn.cursor()

    # Create employee_master table
    cursor.execute('''
        CREATE TABLE employee_master (
            Employee_ID VARCHAR(10),
            Main_Department TEXT,
            Sub_Department TEXT,
            Employee_Name TEXT,
            Designation TEXT,
            Supervisor_Name TEXT,
            MD TEXT,
            PRIMARY KEY (Employee_ID)
        )
    ''')

    # Create emp_util_data table
    cursor.execute('''
        CREATE TABLE emp_util_data (
            Employee_ID VARCHAR(10),
            Week_Ending_Date DATE,
            Month TEXT,
            Charged_Hours DOUBLE,
            Standard_Hours DOUBLE,
            Non_Chargeables DOUBLE,
            FOREIGN KEY (Employee_ID) REFERENCES employee_master(Employee_ID)
        )
    ''')

    # Create emp_plan_hours table
    cursor.execute('''
        CREATE TABLE emp_plan_hours (
            Employee_ID VARCHAR(10),
            
            Week_Ending_Date DATE,
            Month TEXT,
            Planned_Hours DOUBLE,
            FOREIGN KEY (Employee_ID) REFERENCES employee_master(Employee_ID)
        )
    ''')

    # Create emp_skills table
    cursor.execute('''
        CREATE TABLE emp_skills (
            Employee_ID VARCHAR(10),
            Skills TEXT,
            Trainings_Attended TEXT,
            Training_Hours DOUBLE,
            Training_Completion_Month DATE,
            FOREIGN KEY (Employee_ID) REFERENCES employee_master(Employee_ID)
        )
    ''')

    conn.commit()
    conn.close()

# Create tables and insert data
create_tables()

