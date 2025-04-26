import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()


def generate_employee_data(num_employees=500):
    """Генерация реалистичных данных о сотрудниках"""
    np.random.seed(42)
    random.seed(42)

    departments = ['HR', 'IT', 'Finance', 'Marketing', 'Operations', 'Sales', 'R&D']
    positions = ['Junior', 'Middle', 'Senior', 'Team Lead', 'Manager', 'Director']
    educations = ['High School', 'College', 'Bachelor', 'Master', 'PhD']
    performance_levels = ['Low', 'Medium', 'High', 'Top Performer']

    data = []

    for _ in range(num_employees):
        hire_date = fake.date_between(start_date='-10y', end_date='today')
        experience = (datetime.now().date() - hire_date).days // 365

        # Базовые характеристики
        employee = {
            'employee_id': fake.unique.random_number(digits=6),
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'hire_date': hire_date,
            'department': random.choice(departments),
            'position': random.choice(positions),
            'education': random.choice(educations),
            'gender': random.choice(['Male', 'Female']),
            'age': random.randint(22, 65),
            'experience': max(1, min(experience, 20)),
            'salary': 0,  # Будет рассчитано позже
            'performance_score': random.randint(60, 100),
            'performance_level': None,  # Будет рассчитано позже
            'satisfaction_score': random.randint(1, 10),
            'overtime': random.choice(['Yes', 'No']),
            'attrition': random.choice(['Yes', 'No']),
            'training_hours': random.randint(0, 40),
            'last_promotion_date': fake.date_between(start_date=hire_date, end_date='today'),
            'manager_id': None,
            'address': fake.address().replace('\n', ', ')
        }

        # Рассчитываем зарплату на основе позиции и опыта
        base_salary = {
            'Junior': 30000,
            'Middle': 50000,
            'Senior': 80000,
            'Team Lead': 100000,
            'Manager': 120000,
            'Director': 150000
        }[employee['position']]

        experience_bonus = employee['experience'] * 1000
        education_bonus = {
            'High School': 0,
            'College': 2000,
            'Bachelor': 5000,
            'Master': 8000,
            'PhD': 12000
        }[employee['education']]

        employee['salary'] = base_salary + experience_bonus + education_bonus + random.randint(-5000, 5000)

        # Определяем уровень производительности
        if employee['performance_score'] >= 90:
            employee['performance_level'] = 'Top Performer'
        elif employee['performance_score'] >= 75:
            employee['performance_level'] = 'High'
        elif employee['performance_score'] >= 60:
            employee['performance_level'] = 'Medium'
        else:
            employee['performance_level'] = 'Low'

        data.append(employee)

    # Добавляем менеджеров
    managers = [e for e in data if e['position'] in ['Team Lead', 'Manager', 'Director']]
    for employee in data:
        if employee['position'] not in ['Director']:
            employee['manager_id'] = random.choice(managers)['employee_id'] if managers else None

    df = pd.DataFrame(data)

    # Преобразуем даты в строки для CSV
    df['hire_date'] = df['hire_date'].astype(str)
    df['last_promotion_date'] = df['last_promotion_date'].astype(str)

    return df


def save_to_csv(df, filename='employee_data.csv'):
    """Сохранение данных в CSV файл"""
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Данные сохранены в {filename}")


if __name__ == "__main__":
    print("Генерация данных о сотрудниках...")
    employees = generate_employee_data(500)
    save_to_csv(employees, 'employee_data.csv')
    print("Генерация завершена.")