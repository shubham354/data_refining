import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

print("\nOriginal Dataset Shape:", df.shape)

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

print("\nAge range before cleaning:", df['Age'].min(), "to", df['Age'].max())
df = df[df['Age'] >= 0]  
df = df[df['Age'] <= 100]  
print("Age range after cleaning:", df['Age'].min(), "to", df['Age'].max())

duplicates = df.duplicated(subset=['PatientId', 'AppointmentDay'], keep=False)
print("\nNumber of duplicate appointments:", duplicates.sum())

df['Gender'] = df['Gender'].str.upper()
df['No-show'] = df['No-show'].str.upper()

df = df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'})

boolean_columns = ['Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']
for col in boolean_columns:
    df[col] = df[col].astype(bool)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nValue counts for categorical variables:")
for col in ['Gender', 'No-show']:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\nCleaned Dataset Shape:", df.shape)
print("\nCleaned Dataset Info:")
print(df.info())

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=30)
plt.title('Distribution of Patient Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()

conditions = ['Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
condition_percentages = df[conditions].mean() * 100

plt.figure(figsize=(10, 6))
condition_percentages.plot(kind='bar')
plt.title('Percentage of Patients with Different Health Conditions')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
noshow_counts = df['No-show'].value_counts()
plt.pie(noshow_counts, labels=noshow_counts.index, autopct='%1.1f%%')
plt.title('Appointment No-show Rate')
plt.show()

df['AppointmentMonth'] = df['AppointmentDay'].dt.to_period('M')
monthly_appointments = df.groupby('AppointmentMonth').size()

plt.figure(figsize=(12, 6))
monthly_appointments.plot(kind='line', marker='o')
plt.title('Number of Appointments Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Appointments')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
noshow_by_day = df.groupby('DayOfWeek')['No-show'].value_counts(normalize=True).unstack()

plt.figure(figsize=(10, 6))
noshow_by_day.plot(kind='bar')
plt.title('No-show Rates by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Percentage')
plt.legend(title='No-show')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_neighborhoods = df['Neighbourhood'].value_counts().head(10)
plt.figure(figsize=(12, 6))
top_neighborhoods.plot(kind='bar')
plt.title('Top 10 Neighborhoods by Number of Appointments')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Appointments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sms_impact = df.groupby('SMS_received')['No-show'].value_counts(normalize=True).unstack()
plt.figure(figsize=(8, 6))
sms_impact.plot(kind='bar')
plt.title('Impact of SMS on No-show Rates')
plt.xlabel('SMS Received')
plt.ylabel('Percentage')
plt.legend(title='No-show')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[conditions].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Health Conditions')
plt.show()

print("\nSummary Statistics:")
print(df.describe())

print("\nNo-show rates by health condition:")
for condition in conditions:
    show_rate = df[df[condition] == 1]['No-show'].value_counts(normalize=True)
    print(f"\n{condition}:")
    print(show_rate)

print("\nDetailed Summary Statistics:")


numerical_cols = ['Age', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']
print("\nNumerical Variables Statistics:")
print(df[numerical_cols].describe())

print("\nMode for Categorical Variables:")
categorical_cols = ['Gender', 'Neighbourhood', 'No-show']
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    count = df[col].value_counts()[mode_value]
    percentage = (count / len(df)) * 100
    print(f"\n{col}:")
    print(f"Most common value: {mode_value}")
    print(f"Count: {count}")
    print(f"Percentage: {percentage:.2f}%")

df['days_difference'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds() / (24 * 60 * 60)

print("\nTime between scheduling and appointment (days):")
print(df['days_difference'].describe())

print("\nCorrelation between numerical variables:")
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Gender', y='Age')
plt.title('Age Distribution by Gender')
plt.show()

df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 70, 100], labels=['0-18', '19-30', '31-50', '51-70', '70+'])
noshow_by_age = df.groupby('age_group')['No-show'].value_counts(normalize=True).unstack()

plt.figure(figsize=(10, 6))
noshow_by_age.plot(kind='bar')
plt.title('No-show Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='No-show')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
