import pandas as pd
from sportsreference.ncaab.teams import Teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tkinter import filedialog as fd
from sklearn import preprocessing
import numpy as np
from sklearn import metrics


# gets ncaa basketball data from online and saves it a csv. This takes about 10 minutes depending on your internet.
# Once it is saved to a csv, it is much faster to read the csv file in then perform machine learning.
# I created this function so that if I need to update the data, I can just use this function and create a new csv.
def get_ncaab_data_online():
    dataset = pd.DataFrame()
    teams = Teams()
    print("Looking up data, this may take a while...\n")
    print(dir(teams))
    for team in teams:

        print(team.abbreviation)

        #dataset = pd.concat([dataset, team.schedule.dataframe_extended])

    dataset.to_csv("dataset.csv")


def train_and_predict():
    dataset = pd.read_csv(fd.askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),)))

    print(dataset.shape)
    print(dataset.head)

    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass

    dataset.dropna().drop_duplicates()

    fields_to_drop = ['away_points', 'home_points', 'date', 'location',
                      'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                      'winning_name', 'home_ranking', 'away_ranking']

    print("Cleaning up data...\n")
    x = dataset.drop(fields_to_drop, 1)
    print(x.shape)
    print(x.head)

    y = dataset[['home_points', 'away_points']].values

    print(x.shape)
    print(x.head)
    print("Training data...\n")

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    np.savetxt("xtest.csv", x_test, delimiter=",")

    parameters = {'bootstrap': False,
                  'min_samples_leaf': 3,
                  'n_estimators': 50,
                  'min_samples_split': 10,
                  'max_features': 'sqrt',
                  'max_depth': 6}
    model = RandomForestRegressor(**parameters)
    model.fit(x_train.fillna(x_train.mean()), y_train)
    print("Predictions: \n")
    print(model.predict(x_test.fillna(x_test.mean())).astype(int), y_test, "\n\n\n\n")

    predictions = np.asarray(model.predict(x_test.fillna(x_test.mean())).astype(int))
    np.savetxt("predictions.csv", predictions, delimiter=",")

    actual = np.asarray(y_test)
    np.savetxt("actual.csv", actual, delimiter=",")


    #### where the bullshit begins brother
    dataset = pd.DataFrame()
    teams = Teams()

    arkansas = teams('ARKANSAS').schedule.dataframe_extended
    missouri = teams('MISSOURI').schedule.dataframe_extended

    dataset = pd.concat([dataset, arkansas, missouri])

    dataset.to_csv("dataset.csv")

    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass

    dataset.dropna().drop_duplicates()

    fields_to_drop = ['away_points', 'home_points', 'date', 'location',
                      'losing_abbr', 'losing_name', 'winner', 'winning_abbr',
                      'winning_name', 'home_ranking', 'away_ranking']

    print("Cleaning up data...\n")
    x = dataset.drop(fields_to_drop, 1)
    print(x.shape)
    print(x.head)

    y = dataset[['home_points', 'away_points']].values

    print(x.shape)
    print(x.head)
    print("Training data...\n")

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    parameters = {'bootstrap': False,
                  'min_samples_leaf': 3,
                  'n_estimators': 50,
                  'min_samples_split': 10,
                  'max_features': 'sqrt',
                  'max_depth': 6}
    model = RandomForestRegressor(**parameters)
    model.fit(x_train.fillna(x_train.mean()), y_train)
    print("Predictions: \n")
    print(model.predict(x_test.fillna(x_test.mean())).astype(int), y_test, "\n\n\n\n")

    predictions = np.asarray(model.predict(x_test.fillna(x_test.mean())).astype(int))
    np.savetxt("predictionsarkvsmis.csv", predictions, delimiter=",")

    actual = np.asarray(y_test)
    np.savetxt("actual.csv", actual, delimiter=",")

    dataset.to_csv("arkvsmiz.csv")
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == '__main__':

    train_and_predict()
