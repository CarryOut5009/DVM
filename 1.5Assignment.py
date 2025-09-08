import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: LOAD YOUR DATASET
# ============================================================================

df = pd.read_csv(r'C:\Users\jacob\Desktop\Anaconda\Data_Visualization_And_Modeling-main\Data_Visualization_And_Modeling-main\Lab\titanic_passengers.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())

# ============================================================================
# STEP 2: DATASET SUMMARY
# ============================================================================

print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)

"""
Dataset Name: Titanic Passenger Data
Source: Historical records from the RMS Titanic passenger manifest
Collection Method: Data was compiled from official passenger records and survivor accounts from the RMS Titanic disaster
Time Period: April 1912
Sample Size: 891 passengers

Features Description:
- PassengerId: Unique identifier for each passenger
- Survived: Whether the passenger survived (0 = No, 1 = Yes)
- Pclass: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- Name: Passenger's name
- Sex: Gender of the passenger
- Age: Age of the passenger in years
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Ticket: Ticket number
- Fare: Fare paid for the ticket in British pounds
- Cabin: Cabin number (many missing values)
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Why This Dataset Interests Me:
This dataset is fascinating because it represents one of the most famous maritime disasters in history. 
It allows us to explore how social factors like class, gender, and age influenced survival rates during 
a life-or-death situation. The data provides insight into the social dynamics and evacuation procedures 
of the early 20th century, making it both historically significant and statistically rich for analysis.
"""

# ============================================================================
# STEP 3: IDENTIFY 5 INTERESTING QUESTIONS
# ============================================================================

print("\n" + "="*60)
print("5 INTERESTING QUESTIONS TO EXPLORE")
print("="*60)

"""
1. What was the overall survival rate, and how did it differ by passenger class?
2. Did gender significantly impact survival chances?
3. What was the age distribution of passengers, and did age affect survival?
4. How did family size (siblings, spouses, parents, children) relate to survival?
5. How did the combination of gender and passenger class together influence survival?
"""

# ============================================================================
# STEP 4: ANSWER THE QUESTIONS USING PANDAS
# ============================================================================

print("\n" + "="*60)
print("QUESTION 1: What was the overall survival rate, and how did it differ by passenger class?")
print("="*60)

# Overall survival rate
overall_survival_rate = df['Survived'].mean()
print(f"Overall survival rate: {overall_survival_rate:.2%}")

# Survival by passenger class
survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_class.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
print("\nSurvival by Passenger Class:")
print(survival_by_class)

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
survival_by_class['Survival Rate'].plot(kind='bar', color=['gold', 'silver', 'orange'])
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['Pclass'].value_counts().sort_index().plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Number of Passengers by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\nANSWER:")
print("The overall survival rate was 38.4%. However, there was a dramatic disparity based on passenger class.")
print("First-class passengers had a 63% survival rate, second-class had 47%, and third-class had only 24%.")
print("This stark difference suggests that socioeconomic status played a significant role in determining")
print("who survived the disaster, likely due to factors like cabin location, access to lifeboats, and")
print("evacuation procedures that prioritized higher-class passengers.")

print("\n" + "="*60)
print("QUESTION 2: Did gender significantly impact survival chances?")
print("="*60)

# Survival by gender
survival_by_gender = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_gender.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
print("Survival by Gender:")
print(survival_by_gender)

# Cross-tabulation
gender_survival_crosstab = pd.crosstab(df['Sex'], df['Survived'], margins=True)
print("\nCross-tabulation of Gender and Survival:")
print(gender_survival_crosstab)

plt.figure(figsize=(8, 6))
survival_by_gender['Survival Rate'].plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

print("\nANSWER:")
print("Gender had an enormous impact on survival chances. Women had a 74% survival rate compared to")
print("men's 19% survival rate. This dramatic difference reflects the 'women and children first' maritime")
print("evacuation protocol that was strictly followed during the Titanic disaster. The nearly 4:1 survival")
print("ratio between women and men demonstrates how social norms of the era prioritized women's safety.")

print("\n" + "="*60)
print("QUESTION 3: What was the age distribution of passengers, and did age affect survival?")
print("="*60)

# Age analysis
print("Age statistics:")
print(df['Age'].describe())

# Handle missing ages
print(f"\nMissing age data: {df['Age'].isnull().sum()} passengers ({df['Age'].isnull().mean():.1%})")

# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-35)', 
                               'Middle Age (36-60)', 'Elderly (60+)'])

# Survival by age group
survival_by_age = df.groupby('Age_Group')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_age.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
print("\nSurvival by Age Group:")
print(survival_by_age)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df['Age'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.subplot(1, 2, 2)
survival_by_age['Survival Rate'].plot(kind='bar', color='lightgreen')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\nANSWER:")
print("The age distribution shows most passengers were young adults, with a median age of 28 years.")
print("Age did affect survival rates, with children (0-12) having the highest survival rate at 58%,")
print("followed by elderly passengers at 43%. Young adults and middle-aged passengers had lower")
print("survival rates around 36-38%. This pattern supports the 'women and children first' protocol")
print("and suggests that very young passengers were given priority during evacuation.")

print("\n" + "="*60)
print("QUESTION 4: How did family size (siblings, spouses, parents, children) relate to survival?")
print("="*60)

# Create family size variable
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves

# Analyze family size
print("Family Size Distribution:")
family_size_counts = df['Family_Size'].value_counts().sort_index()
print(family_size_counts)

# Survival by family size
survival_by_family = df.groupby('Family_Size')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_family.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
print("\nSurvival by Family Size:")
print(survival_by_family)

# Create family status categories
df['Family_Status'] = df['Family_Size'].apply(lambda x: 'Alone' if x == 1 
                                             else 'Small Family (2-4)' if x <= 4 
                                             else 'Large Family (5+)')

survival_by_status = df.groupby('Family_Status')['Survived'].agg(['count', 'sum', 'mean'])
survival_by_status.columns = ['Total Passengers', 'Survivors', 'Survival Rate']
print("\nSurvival by Family Status:")
print(survival_by_status)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
survival_by_family['Survival Rate'].plot(kind='bar', color='orange')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')

plt.subplot(1, 2, 2)
survival_by_status['Survival Rate'].plot(kind='bar', color='purple')
plt.title('Survival Rate by Family Status')
plt.xlabel('Family Status')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\nANSWER:")
print("Family size had a complex relationship with survival. Passengers traveling alone had a 30%")
print("survival rate, while those in small families (2-4 people) had the highest survival rate at 58%.")
print("However, passengers in large families (5+ people) had only a 16% survival rate. This suggests")
print("that having some family members provided advantages (mutual help, shared resources), but very")
print("large families may have faced difficulties coordinating evacuation or been reluctant to separate.")

print("\n" + "="*60)
print("QUESTION 5: How did the combination of gender and passenger class together influence survival?")
print("="*60)

# Create crosstab of gender, class, and survival
gender_class_survival = df.groupby(['Sex', 'Pclass'])['Survived'].agg(['count', 'sum', 'mean']).round(3)
gender_class_survival.columns = ['Total', 'Survivors', 'Survival Rate']
print("Survival by Gender and Passenger Class:")
print(gender_class_survival)

# Create a pivot table for better visualization
survival_pivot = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
print("\nSurvival Rate Matrix (Gender x Class):")
print(survival_pivot.round(3))

# Compare specific groups
print("\nSpecific Group Comparisons:")
third_class_women = df[(df['Sex'] == 'female') & (df['Pclass'] == 3)]['Survived'].mean()
first_class_men = df[(df['Sex'] == 'male') & (df['Pclass'] == 1)]['Survived'].mean()
print(f"3rd class women survival rate: {third_class_women:.1%}")
print(f"1st class men survival rate: {first_class_men:.1%}")

# Visualization
plt.figure(figsize=(10, 6))
survival_pivot.plot(kind='bar', color=['gold', 'silver', 'orange'])
plt.title('Survival Rate by Gender and Passenger Class')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.legend(title='Passenger Class', labels=['1st Class', '2nd Class', '3rd Class'])
plt.xticks(rotation=0)
plt.show()

print("\nANSWER:")
print("The combination of gender and class created a hierarchy of survival chances. First-class women")
print("had the highest survival rate at 97%, while third-class men had the lowest at 14%. Remarkably,")
print("third-class women (50% survival rate) had better chances than first-class men (37% survival rate),")
print("demonstrating that gender was actually a stronger predictor of survival than social class. This")
print("shows how the 'women and children first' protocol transcended class boundaries, though class")
print("still provided additional advantages within each gender group.")