import pandas as pd

data = pd.read_csv('test_data/DE/combined_test.csv', delimiter=';')
print('All Wierschem addresses:')
print(data[data['cOrtsname'].str.contains('Wierschem', case=False, na=False)])
print('\nAll addresses with Spreeg:')
print(data[data['cStrassenname'].str.contains('Spreeg', case=False, na=False)])
