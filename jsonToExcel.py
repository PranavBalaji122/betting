import pandas as pd

# Data provided by the user
data_pts = {
    'Drew Peterson': 6.7622, 'Rob Dillingham': 8.9594, 'Wenyen Gabriel': 4.259, 'Dario Šarić': 7.3772666666666655,
    'Steven Adams': 4.3732, 'Larry Nance Jr.': 7.299966666666667, 'Jules Bernard': 4.11645, 'Jae Crowder': 3.379833333333333,
    'Robert Covington': 4.7600333333333325, 'Doug McDermott': 5.5024, 'Cole Swider': 6.8322, 'Tosan Evbuomwan': 8.692533333333333,
    'Talen Horton-Tucker': 6.0571, 'Tristan Vukcevic': 15.1064, 'Monte Morris': 4.7875, 'Shai Gilgeous-Alexander': 31.6769,
    'Tre Mann': 10.577763333333335, 'Joe Harris': 7.9569, 'KJ Simpson': 6.1706, 'Isaac Jones': 7.9851, 'Jordan McLaughlin': 4.2588,
    'Armoni Brooks': 11.0913, 'Markieff Morris': 2.7365, 'Luka Dončić': 33.136, 'Lindy Waters III': 4.7674, 'Jayson Tatum': 27.1728,
    'Kevin Durant': 28.1498, 'Jamal Shead': 6.4721, 'LeBron James': 26.0052, 'Ja Morant': 24.7368, 'Dillon Jones': 4.0165,
    'Matthew Hurt': 6.6538, 'Pascal Siakam': 19.0659, 'Patty Mills': 3.5315, 'RJ Barrett': 23.4612, 'Jalen Pickett': 2.6779,
    'Julius Randle': 24.284, 'Darius Bazley': 9.1308, 'Mike Muscala': 3.1797, 'Giannis Antetokounmpo': 32.2398, 'Kawhi Leonard': 23.5179,
    'Tyler Herro': 18.9886, 'Jordan Miller': 8.1011, 'Nikola Jokić': 27.5787, 'Cameron Payne': 4.8977, 'DeMar DeRozan': 24.8326,
    'Marvin Bagley III': 10.2856, 'Kyle Anderson': 7.1794, "Day'Ron Sharpe": 'DNW', 'Domantas Sabonis': 17.7259
}

data_trb = {
    'Mãozinha Pereira': 8.593, 'Richaun Holmes': 4.2238, 'Mike Muscala': 3.0566, 'Monte Morris': 0.5007,
    'P.J. Tucker': 3.7608, 'Isaiah Mobley': 1.9202, 'Jordan Walsh': 0.6229, 'Otto Porter Jr.': 2.3902,
    'Jaylen Nowell': 1.48825, 'Darius Bazley': 5.2431, 'Jordan McLaughlin': 1.3966, 'Kobe Bufkin': 2.373,
    'Ibou Badji': 4.094, 'Karl-Anthony Towns': 8.1776, 'Thaddeus Young': 3.7219, 'Colby Jones': 1.2531,
    'Sidy Cissoko': 3.7233, 'Zeke Nnaji': 2.2356, 'Markieff Morris': 1.8845, 'Domantas Sabonis': 12.3006,
    'Dejounte Murray': 5.7561, 'DeAndre Jordan': 3.67735, 'Mouhamadou Gueye': 1.7734, 'Jason Preston': 4.1342,
    'Joe Harris': 1.5801, 'Shake Milton': 1.1325, 'Justin Champagnie': 7.6973, 'Anthony Davis': 13.3052,
    'Mouhamed Gueye': 7.4907, 'Jalen Smith': 5.337766666666667, 'Isaiah Hartenstein': 12.9097, 'Joel Embiid': 11.3801,
    'David Roddy': 4.4712, 'Gui Santos': 1.171, 'Nikola Jokić': 12.3709, 'Wendell Moore Jr.': 2.2236, 'Giannis Antetokounmpo': 11.2707,
    'Nikola Vučević': 8.7179, 'Cody Zeller': 2.9939, 'Kyle Filipowski': 5.579366666666667, 'Justin Minaya': 2.10985,
    'Isaac Jones': 2.69285, 'Orlando Robinson': 4.044933333333333, 'Luka Dončić': 8.7867, 'Andre Drummond': 12.3861,
    'Marques Bolden': 4.543, 'Clint Capela': 11.3183, 'Rudy Gobert': 13.2101, 'Dario Šarić': 5.449533333333333
}

data_ast = {
    'Patty Mills': 0.568, 'Kenneth Lofton Jr.': 1.0279, 'Xavier Moon': 3.9553, 'Quenton Jackson': 2.981,
    'Isaiah Hartenstein': 3.1241, 'Dillon Jones': 1.4649, 'Danilo Gallinari': 1.0096, 'Tyler Kolek': 0.6941,
    'Trae Young': 12.2056, 'A.J. Lawson': 0.0701, 'David Roddy': 1.3811, 'Markieff Morris': 0.0052, 'Quentin Grimes': 1.2142,
    'Cade Cunningham': 8.1978, 'Tre Mann': 4.9759, 'LaMelo Ball': 7.1425, 'Ja Morant': 8.4889, 'Kira Lewis Jr.': 0.906,
    'Chris Paul': 5.49995, 'LeBron James': 8.767, 'James Harden': 8.514, 'Shai Gilgeous-Alexander': 5.3863,
    'Darius Garland': 6.1543, 'Josh Giddey': 3.9523, 'Tyus Jones': 11.5084, 'Luka Dončić': 7.8579, 'Nikola Jokić': 8.5959,
    'Talen Horton-Tucker': 4.6911, 'Julius Randle': 4.1376, 'Scottie Barnes': 6.7307, 'Tyrese Haliburton': 10.83535,
    'DeJon Jarreau': 1.8809, 'Ben Simmons': 6.8964, 'Devin Booker': 7.0921, 'Jalen Brunson': 6.5935,
    'Domantas Sabonis': 7.5693, 'Damian Lillard': 7.1974, 'Brandon Ingram': 5.4326, 'Luka Šamanić': 0.173,
    'Seth Curry': 0.7247, 'Jacob Gilyard': 3.7989, 'Cody Zeller': 1.3232, 'Kyle Anderson': 3.6447, 'Jamal Murray': 7.0423,
    'Mike Muscala': 0.1611, 'Draymond Green': 5.29475, 'DeMar DeRozan': 4.4593, 'Trent Forrest': 2.5903,
    'Dennis Schröder': 7.456583333333333, 'Cameron Payne': 0.8621
}

# Create DataFrame from the dictionaries
df_pts = pd.DataFrame(list(data_pts.items()), columns=['Player', 'Points'])
df_trb = pd.DataFrame(list(data_trb.items()), columns=['Player', 'Rebounds'])
df_ast = pd.DataFrame(list(data_ast.items()), columns=['Player', 'Assists'])

# Merge all data into one DataFrame
df_combined = pd.merge(df_pts, df_trb, on='Player', how='outer')
df_combined = pd.merge(df_combined, df_ast, on='Player', how='outer')

# Save to Excel
file_path = '/Users/pranavbalaji/Desktop/nba_player_stats.xlsx'
df_combined.to_excel(file_path, index=False)

file_path  # Return file path for downloading
