import sqlite3
import pandas as pd
# Export table values to Excel sheets
conn = sqlite3.connect('employee.db')

# Export employee_master table
df_employee_master = pd.read_sql_query("SELECT * FROM employee_master", conn)
df_employee_master.to_excel('employee_master.xlsx', index=False)

# Export emp_util_data table
df_emp_util_data = pd.read_sql_query("SELECT * FROM emp_util_data", conn)
df_emp_util_data.to_excel('emp_util_data.xlsx', index=False)

# Export emp_plan_hours table
df_emp_plan_hours = pd.read_sql_query("SELECT * FROM emp_plan_hours", conn)
df_emp_plan_hours.to_excel('emp_plan_hours.xlsx', index=False)

# Export emp_skills table
df_emp_skills = pd.read_sql_query("SELECT * FROM emp_skills", conn)
df_emp_skills.to_excel('emp_skills.xlsx', index=False)

conn.close()
