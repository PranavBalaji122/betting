from randomForestRegPredictor import run
from consienstyTest import getConsistency


def runner():
    data_player_pts = getConsistency('pts')[0]
    data_team_pts = getConsistency('pts')[1]
    data_player_trb = getConsistency('trb')[0]
    data_team_trb = getConsistency('trb')[1]
    data_player_ast = getConsistency('ast')[0]
    data_team_ast = getConsistency('ast')[1]
    teams_predtcing = ['LAL', 'SAC', 'MIA', 'ORL', 'MEM', 'ATL', 'UTA', 'BRK', 'NYK'
                       'NOP', 'WAS', 'MIL', 'GSW', 'MIN', 'PHI', 'CLE', 'BOS'
                       'CHI', 'POR', 'SAS', 'LAC', 'DAL', 'DET', 'PHO']
    filtered_team_pts = []
    filtered_player_pts = []
    filtered_team_trb = []
    filtered_player_trb = []
    filtered_team_ast = []
    filtered_player_ast = []

    for i in range(len(data_team_pts)):
        if data_team_pts[i] in teams_predtcing:
            filtered_team_pts.append(data_team_pts[i])
            filtered_player_pts.append(data_player_pts[i])
    for i in range(len(data_team_trb)):
        if data_team_trb[i] in teams_predtcing:
            filtered_team_trb.append(data_team_trb[i])
            filtered_player_trb.append(data_player_trb[i])
    for i in range(len(data_team_ast)):
        if data_team_ast[i] in teams_predtcing:
            filtered_team_ast.append(data_team_ast[i])
            filtered_player_ast.append(data_player_ast[i])
    


    output_pts = {}
    output_trb = {}
    output_ast = {}

    # Loop for points prediction
    for i in range(len(data_player_pts)):
        try:
            output_pts[data_player_pts[i]] = run(data_player_pts[i], data_team_pts[i], 'pts')
        except Exception as e:
            print(f"Error occurred for player {data_player_pts[i]}: {e}")
            output_pts[data_player_pts[i]] = "DNW"

    # Loop for rebounds prediction
    for i in range(len(data_player_trb)):
        try:
            output_trb[data_player_trb[i]] = run(data_player_trb[i], data_team_trb[i], 'trb')
        except Exception as e:
            print(f"Error occurred for player {data_player_trb[i]}: {e}")
            output_trb[data_player_trb[i]] = "DNW"

    # Loop for assists prediction
    for i in range(len(data_player_ast)):
        try:
            output_ast[data_player_ast[i]] = run(data_player_ast[i], data_team_ast[i], 'ast')
        except Exception as e:
            print(f"Error occurred for player {data_player_ast[i]}: {e}")
            output_ast[data_player_ast[i]] = "DNW"

    
    print(output_pts)
    print(output_trb)
    print(output_ast)


if __name__ == '__main__':
    runner()
