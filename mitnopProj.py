import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

file_path = 'C:/Users/programiranje/Desktop/MitnopProjekatSrcanaBolest/dataset.csv'

data = pd.read_csv(file_path)

# prikaz prvih 10 vrednosti
print("Prvih 10 vrednosti:")
print(data.head(10))

# prikaz poslednjih 10 vrednosti
print("Poslednjih 10 vrednosti:")
print(data.tail(10))

# prikaz tipova kolona
print("Tipovi kolona:")
print(data.dtypes)

# uklanjanje vrednosti
data.dropna(inplace=True)

# mapiranje 
data['Heart_Disease'] = data['Heart_Disease'].map({'Absence': 0, 'Presence': 1})

categorical_columns = data.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('Heart_Disease', errors='ignore')  

for column in categorical_columns:
    data[column] = data[column].astype('category').cat.codes

# provera tipova kolona
print("Tipovi kolona novi:")
print(data.dtypes)

# podela godina
data['Age_Group'] = pd.cut(data['Age'], bins=[25, 40, 55, 70, 80], labels=['25-40', '40-55', '55-70', '70-80'])


#-------------------------Oboleli po starosnoj grupi----------------------------------------------#
# histogram za svaku grupu
data['Heart_Disease'] = data['Heart_Disease'].replace({0: 'Absence', 1: 'Presence'})

plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='Age_Group', hue='Heart_Disease', multiple='stack', palette='Set1')
plt.title('Distribucija srčanih bolesti po starosnoj grupi')
plt.xlabel('Starosna grupa')
plt.ylabel('Broj')

plt.legend(title='Srčane bolesti', labels=['Bez srčanih bolesti', 'Sa srčanim bolestima'])
plt.show()


grouped_data = data.groupby(['Age_Group', 'Sex', 'Heart_Disease']).size().unstack(fill_value=0)
heart_disease_percent = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100
print("Procenat srčanih oboljenja po starosnim grupama i polu:")
print(heart_disease_percent)

# grua sa najvisim procentom oboljenja
most_susceptible_age_group = heart_disease_percent.idxmax(axis=0)
print("\nStarosne grupe sa najvišim procentom srčanih oboljenja:")
print(most_susceptible_age_group)

#-----------------------------KORELACIJA---------------------------------------------#
# mapiranje 
data['Heart_Disease'] = data['Heart_Disease'].map({'Absence': 0, 'Presence': 1})

# korelacija između godina i prisustva srčanih oboljenja
correlation_age_heart_disease = data['Age'].corr(data['Heart_Disease'])
threshold = 0.1

if abs(correlation_age_heart_disease) > threshold:
    print("Godine imaju uticaj na prisustvo srčanih oboljenja.")
else:
    print("Godine nemaju značajan uticaj na prisustvo srčanih oboljenja.")

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

#--------------Grupisano po polu, starosnoj grupi i prisustu oboljenja----------------#
grouped_data = data.groupby(['Age_Group', 'Sex', 'Heart_Disease']).size().unstack(fill_value=0)

# podataci za muskarce
male_data = grouped_data.loc[(slice(None), 1), :]

plt.figure(figsize=(12, 6))
male_data.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], rot=0)
plt.title('Distribucija Srčanih Bolesti kod Muškaraca po Starosnoj Grupi')
plt.xlabel('Starosna Grupa, Muškarci')
plt.ylabel('Broj')
plt.legend(title='Srčane Bolesti', labels=['Bez Srčanih Bolesti', 'Sa Srčanim Bolestima'])
plt.show()

# podataci za žene
female_data = grouped_data.loc[(slice(None), 0), :]

plt.figure(figsize=(12, 6))
female_data.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], rot=0)
plt.title('Distribucija Srčanih Bolesti kod Žena po Starosnoj Grupi')
plt.xlabel('Starosna Grupa, Žene')
plt.ylabel('Broj')
plt.legend(title='Srčane Bolesti', labels=['Bez Srčanih Bolesti', 'Sa Srčanim Bolestima'])
plt.show()

#-------------------Odnos holesterola, BP i EKG-a kod muskaraca i zena-----------------#

grouped_sex_data = data.groupby('Sex').agg({'Cholesterol': 'mean', 'BP': 'mean', 'EKG_results': 'mean'})

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
grouped_sex_data['Cholesterol'].plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=None)
plt.title('Holesterol po Polu')
plt.legend(labels=['Muški', 'Ženski'])
plt.ylabel('')

plt.subplot(1, 3, 2)
grouped_sex_data['BP'].plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=None)
plt.title('Krvni Pritisak po Polu')
plt.legend(labels=['Muški', 'Ženski'])
plt.ylabel('')

plt.subplot(1, 3, 3)
grouped_sex_data['EKG_results'].plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=None)
plt.title('EKG Rezultati po Polu')
plt.legend(labels=['Muški', 'Ženski'])
plt.ylabel('')

plt.tight_layout()
plt.show()

#-------------Logisticka regresija da li povecanje par. poveca mogucnost za srcano oboljenje -----------------------#

def logistic_regression(data, gender):
    X = data[['Cholesterol', 'BP', 'EKG_results', 'Max_HR']]
    y = data['Heart_Disease'].astype(int)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = sm.add_constant(X_scaled)

    
    model = sm.Logit(y, X_scaled)
    result = model.fit()

    #rezultati
    result_summary = result.summary()
    print(f"\nLogistička regresija za {gender}:\n")
    print(result_summary)

    # ispis rezultata
    interpretation = {}
    for param, coef in zip(['const', 'Cholesterol', 'BP', 'EKG_results', 'Max_Heart_Rate'], result.params):
        if param != 'const':
            if coef > 0:
                interpretation[param] = "Povećanje povećava verovatnoću prisustva srčane bolesti."
            else:
                interpretation[param] = "Povećanje smanjuje verovatnoću prisustva srčane bolesti."

    for param, desc in interpretation.items():
        print(f"{param}: {desc}")

# provera
if 'Max_HR' in data.columns:
    # Logistička regresija za muškarce
    logistic_regression(data[data['Sex'] == 1], 'muškarce')

    # Logistička regresija za žene
    logistic_regression(data[data['Sex'] == 0], 'žene')
else:
    print("Kolona 'Max_Heart_Rate' ne postoji u setu podataka.")

#---------------------Potvrda istrazivanja---------------------------------------#

age_groups = data.groupby('Age_Group')

# prosečni holesterola i krvni pritisak po starosnim grupama
cholesterol_by_age = age_groups['Cholesterol'].mean()
bp_by_age = age_groups['BP'].mean()

# vizualizacija odnosa godina, holesterola i krvnog pritiska
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
sns.lineplot(x=cholesterol_by_age.index, y=cholesterol_by_age.values)
plt.title('Holesterol po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečan Holesterol')

ax2 = plt.subplot(1, 2, 2)
sns.lineplot(x=bp_by_age.index, y=bp_by_age.values)
plt.title('Krvni Pritisak po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečan Krvni Pritisak')

plt.tight_layout()
plt.show()

#------------------------ Odnosa godina, EKG-a i maksimalnog broja otkucaja srca-------------------#
ecg_by_age = age_groups['EKG_results'].mean()
hr_by_age = age_groups['Max_HR'].mean()

plt.figure(figsize=(12, 6))

ax1 = plt.subplot(1, 2, 1)
sns.lineplot(x=ecg_by_age.index, y=ecg_by_age.values)
plt.title('EKG po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečna Vrednost EKG')

ax2 = plt.subplot(1, 2, 2)
sns.lineplot(x=hr_by_age.index, y=hr_by_age.values)
plt.title('Maksimalan Broj Otkucaja Srca po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečan Maksimalan Broj Otkucaja Srca')

plt.tight_layout()
plt.show()
#zavisnost rezultata ekg i jacine bola u grudima 
chest_pain_types = data['Chest_pain_type'].unique()
avg_ekg_results = [data[data['Chest_pain_type'] == t]['EKG_results'].mean() for t in chest_pain_types]

# stubicasti grafikon
plt.figure(figsize=(8, 6))
plt.bar(chest_pain_types, avg_ekg_results)

plt.xlabel('Јаčina bola u grudima')
plt.ylabel('Prosečni rezultati EKG-a')
plt.title('Prosečni rezultati EKG-a po jačini bola u grudima')

plt.grid()

plt.show()


#---------------------Uticaj broja krvnih sudova na krvni pritisak u odnosu na bol i starosnu grupu--------#

vessels_by_age = data.groupby('Age')['Number_of_vessels_fluro'].mean()
bp_by_age = data.groupby('Age')['BP'].mean()

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
sns.lineplot(x=vessels_by_age.index, y=vessels_by_age.values)
plt.title('Broj Krvnih Sudova po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečan Broj Krvnih Sudova')

# krvni pritisak
ax2 = plt.subplot(1, 2, 2)
sns.lineplot(x=bp_by_age.index, y=bp_by_age.values)
plt.title('Krvni Pritisak po Starosnoj Grupi')
plt.xlabel('Starosna Grupa')
plt.ylabel('Prosečan Krvni Pritisak')

plt.tight_layout()
plt.show()

# --------------------------Prikaz kolona u zavisnosti da li ima srcano oboljenje ili ne-----------------#
grouped_data = data.groupby('Heart_Disease')

heart_disease_patients = grouped_data.get_group(1)

healthy_patients = grouped_data.get_group(0)

plt.figure(figsize=(8, 6))
plt.hist(heart_disease_patients['BP'], bins=10, color='red', alpha=0.5, label='Sa srčanom bolešću')
plt.hist(healthy_patients['BP'], bins=10, color='green', alpha=0.5, label='Bez srčane bolesti')
plt.xlabel('Krvni pritisak')
plt.ylabel('Broj pacijenata')
plt.title('Distribucija krvnog pritiska - Sa i bez srčane bolesti')
plt.legend()
plt.show()

# grafik za holesterol
plt.figure(figsize=(8, 6))
plt.hist(heart_disease_patients['Cholesterol'], bins=10, color='red', alpha=0.5, label='Sa srčanom bolešću')
plt.hist(healthy_patients['Cholesterol'], bins=10, color='green', alpha=0.5, label='Bez srčane bolesti')
plt.xlabel('Holesterol')
plt.ylabel('Broj pacijenata')
plt.title('Distribucija holesterola - Sa i bez srčane bolesti')
plt.legend()
plt.show()

# grafik za maksimalni broj otkucaja srca
plt.figure(figsize=(8, 6))
plt.hist(heart_disease_patients['Max_HR'], bins=10, color='red', alpha=0.5, label='Sa srčanom bolešću')
plt.hist(healthy_patients['Max_HR'], bins=10, color='green', alpha=0.5, label='Bez srčane bolesti')
plt.xlabel('Maksimalni broj otkucaja srca')
plt.ylabel('Broj pacijenata')
plt.title('Distribucija maksimalnog broja otkucaja srca - Sa i bez srčane bolesti')
plt.legend()
plt.show()
#---------------------------------------------------------------------------------------------------#
#-------------------------------------------MODELI--------------------------------------------------#
#---------------------------------------------------------------------------------------------------#

file_path = 'C:/Users/programiranje/Desktop/MitnopProjekatSrcanaBolest/dataset.csv'

data = pd.read_csv(file_path)


data.dropna(inplace=True)

#mMapiranje 
data['Heart_Disease'] = data['Heart_Disease'].map({'Absence': 0, 'Presence': 1})
# podela podataka na obelezja (X) i ciljnu promenljivu (y)
X = data.drop(['Heart_Disease'], axis=1)
y = data['Heart_Disease']

# podela podataka na trening i test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# inicijalizacija modela
logreg_model = LogisticRegression(solver='liblinear', penalty='l1')
dt_model = DecisionTreeClassifier()
svm_model = SVC()

# Grid pretraga za logisticku regresiju
logreg_param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
logreg_grid_search = GridSearchCV(estimator=logreg_model, param_grid=logreg_param_grid, cv=5)
logreg_grid_search.fit(X_train, y_train)
best_logreg_model = logreg_grid_search.best_estimator_

# Grid pretraga za drvo odlucivanja
dt_param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid, cv=5)
dt_grid_search.fit(X_train, y_train)
best_dt_model = dt_grid_search.best_estimator_

# Grid pretraga za SVM
svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)
best_svm_model = svm_grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Matrica konfuzije
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Set Evaluation ({model_name}):")
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.show()
    
    return accuracy, precision, recall, f1

# evaluacija modela
logreg_metrics = evaluate_model(best_logreg_model, X_test, y_test, "Logistic Regression")
print()
dt_metrics = evaluate_model(best_dt_model, X_test, y_test, "Decision Tree")
print()
svm_metrics = evaluate_model(best_svm_model, X_test, y_test, "Support Vector Machine")
print()


metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'SVM'],
    'Accuracy': [logreg_metrics[0], dt_metrics[0], svm_metrics[0]],
    'Precision': [logreg_metrics[1], dt_metrics[1], svm_metrics[1]],
    'Recall': [logreg_metrics[2], dt_metrics[2], svm_metrics[2]],
    'F1 Score': [logreg_metrics[3], dt_metrics[3], svm_metrics[3]]
})

metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Comparison of Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.show()

# koeficijenati logisticke regresije
coefficients = best_logreg_model.coef_
intercept = best_logreg_model.intercept_

# prikazivanje koeficijenata
feature_names = X.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})
coef_df['Coefficient'] = coef_df['Coefficient'].abs()  # Apsolutne vrednosti koeficijenata za bolje poređenje
coef_df['Coefficient'] = coef_df['Coefficient'].round(3)
coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
print(coef_df)


# prikazivanje vaznosti atributa
data.rename(columns={
    'Heart_Disease': 'Srčana_Bolest',
    'Thallium': 'Talijum',
    'Number_of_vessels_fluro': 'Broj_kvnih_sudova_fluro',
    'Exercise_angina': 'Angina_pri_vežbanju',
    'Max_HR': 'Maks_Broj_Otkucaja_Srca',
    'ST_depression': 'ST_Depresija',
    'Chest_pain_type': 'Tip_Bola_u_Grudima',
    'Slope_of_ST': 'Nagib_ST',
    'Sex': 'Pol',
    'Age': 'Godine',
    'EKG_results': 'EKG_Rezultati',
    'BP': 'Krvni_Pritisak',
    'Cholesterol': 'Holesterol',
    'FBS_over_120': 'FBS_preko_120'
}, inplace=True)

coef_df['Feature'] = coef_df['Feature'].map({
    'Thallium': 'Talijum',
    'Number_of_vessels_fluro': 'Broj_kvnih_sudova_fluro',
    'Exercise_angina': 'Angina_pri_vežbanju',
    'Max_HR': 'Maks_Broj_Otkucaja_Srca',
    'ST_depression': 'ST_Depresija',
    'Chest_pain_type': 'Tip_Bola_u_Grudima',
    'Slope_of_ST': 'Nagib_ST',
    'Sex': 'Pol',
    'Age': 'Godine',
    'EKG_results': 'EKG_Rezultati',
    'BP': 'Krvni_Pritisak',
    'Cholesterol': 'Holesterol',
    'FBS_over_120': 'FBS_preko_120'
})

plt.figure(figsize=(10, 6))
plt.bar(coef_df['Feature'], coef_df['Coefficient'])
plt.xticks(rotation=90)
plt.xlabel('Atribut')
plt.ylabel('Koeficijent')
plt.title('Važnost Atributa')
plt.show()

