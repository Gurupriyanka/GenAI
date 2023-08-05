# List of employees with their details
employees = [
    ('EMP001', 'ABC', 'Sales', 'John Doe', 'Manager', 'Supervisor1', 'MD1'),
    ('EMP002', 'ABC', 'Sales', 'Jane Smith', 'Analyst', 'Supervisor1', 'MD1'),
    ('EMP003', 'ABC', 'Marketing', 'Michael Johnson', 'Developer', 'Supervisor1', 'MD1'),
    ('EMP004', 'ABC', 'Marketing', 'Emily Davis', 'Data Scientist', 'Supervisor1', 'MD1'),
    ('EMP005', 'ABC', 'Tax', 'David Wilson', 'Engineer', 'Supervisor1', 'MD1'),
    ('EMP006', 'DEF', 'Sales', 'Sarah Anderson', 'Manager', 'Supervisor2', 'MD1'),
    ('EMP007', 'DEF', 'Sales', 'James Johnson', 'Analyst', 'Supervisor2', 'MD1'),
    ('EMP008', 'DEF', 'Marketing', 'Emma Thompson', 'Developer', 'Supervisor2', 'MD1'),
    ('EMP009', 'DEF', 'Marketing', 'Oliver Clark', 'Data Scientist', 'Supervisor2', 'MD1'),
    ('EMP010', 'DEF', 'Tax', 'Sophia Adams', 'Engineer', 'Supervisor2', 'MD1'),
    ('EMP011', 'GHI', 'Sales', 'William Turner', 'Manager', 'Supervisor3', 'MD2'),
    ('EMP012', 'GHI', 'Sales', 'Lily Roberts', 'Analyst', 'Supervisor3', 'MD2'),
    ('EMP013', 'GHI', 'Marketing', 'Benjamin Walker', 'Developer', 'Supervisor3', 'MD2'),
    ('EMP014', 'GHI', 'Marketing', 'Ava Moore', 'Data Scientist', 'Supervisor3', 'MD2'),
    ('EMP015', 'GHI', 'Tax', 'Henry Hughes', 'Engineer', 'Supervisor3', 'MD2')
]

# Generate and print the insert statements
for employee in employees:
    insert_statement = "INSERT INTO employee_master (Employee_ID, Main_Department, Sub_Department, Employee_Name, Designation, Supervisor_Name, MD) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(*employee)
    print(insert_statement)
