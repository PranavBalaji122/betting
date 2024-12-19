from randomForestRegPredictor import run



def runner():
    player_names = [
    "Jalen Suggs", "Isaiah Hartenstein", "Luguentz Dort", "Alex Caruso", "Collin Sexton", 
    "Jordan Clarkson", "Keyonte George", "Jalen Duren", "Walker Kessler", "Jayson Tatum", 
    "Coby White", "Derrick White", "Ayo Dosunmu", "Patrick Williams", "Trae Young", 
    "Jalen Johnson", "De'Andre Hunter", "Devin Vassell", "Jeremy Sochan", "Dyson Daniels", 
    "Harrison Barnes", "Stephen Curry", "Jonathan Kuminga", "Andrew Wiggins", "Desmond Bane", 
    "Dejounte Murray", "Norman Powell", "Klay Thompson", "Amir Coffey", "Dereck Lively II", 
    "Devin Booker", "Kevin Durant", "Anthony Edwards", "Mikal Bridges", "OG Anunoby", 
    "Josh Hart", "Rudy Gobert", "Anfernee Simons", "Jerami Grant", "Aaron Gordon", 
    "Julian Strawther", "Toumani Camara", "Anthony Davis", "De'Aaron Fox", "DeMar DeRozan", 
    "LeBron James", "Domantas Sabonis", "Rui Hachimura", "Goga Bitadze", "Lauri Markkanen", 
    "Julian Champagnie", "Draymond Green"
    ]
    opponents = [
    "OKC", "ORL", "ORL", "ORL", "DET", 
    "DET", "DET", "UTA", "DET", "CHI", 
    "BOS", "CHI", "BOS", "BOS", "SAS", 
    "SAS", "SAS", "ATL", "ATL", "SAS", 
    "ATL", "MEM", "MEM", "MEM", "GSW", 
    "HOU", "DAL", "LAC", "DAL", "LAC", 
    "IND", "IND", "NYK", "MIN", "MIN", 
    "MIN", "NYK", "DEN", "DEN", "POR", 
    "POR", "DEN", "SAC", "LAL", "LAL", 
    "SAC", "LAL", "SAC", "OKC", "DET", 
    "ATL", "MEM"
    ]
    output = {}
    for i in range(len(opponents)):
        output[player_names[i]] = run(player_names[i],opponents[i],'pts')
        
    print(output)


if __name__ == '__main__':
    runner()
