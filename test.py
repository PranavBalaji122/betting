def add_win_dict(result_dict):
    # Convert the result_dict to a list of tuples: (team_name, score_int)
    team_scores = []
    for team_id, (score_str, team_name) in result_dict.items():
        score_int = int(score_str)
        team_scores.append((team_name, score_int))
    
    # Find all distinct scores in descending order
    distinct_scores = sorted({score for _, score in team_scores}, reverse=True)
    
    # Determine the top three distinct scores
    top_three_scores = set(distinct_scores[:3])
    
    # Build the result dictionary including teams with scores in the top three distinct scores
    result = {}
    for team_name, score in team_scores:
        if score in top_three_scores:
            result[team_name] = score
    return result

assert add_win_dict({'A': ['10', 'Alice'], 'B': ['5', 'Bob'], 'C': ['15', 'Charlie']}) == {'Charlie': 15, 'Alice': 10, 'Bob': 5}
assert add_win_dict({'A': ['10', 'Alice'], 'B': ['10', 'Bob'], 'C': ['10', 'Charlie']}) == {'Alice': 10, 'Bob': 10, 'Charlie': 10}
assert add_win_dict({'A': ['10', 'Alice'], 'B': ['5', 'Bob'], 'C': ['15', 'Charlie'], 'D': ['8', 'David']}) == {'Charlie': 15, 'Alice': 10, 'David': 8}
assert add_win_dict({'A': ['-5', 'Alice'], 'B': ['0', 'Bob'], 'C': ['5', 'Charlie']}) == {'Charlie': 5, 'Bob': 0, 'Alice': -5}
assert add_win_dict({'A': ['100', 'Alice'], 'B': ['200', 'Bob'], 'C': ['300', 'Charlie'], 'D': ['150', 'David'], 'E': ['250', 'Eve']}) == {'Charlie': 300, 'Eve': 250, 'Bob': 200}
assert add_win_dict({'A': ['10', 'Alice'], 'B': ['10', 'Bob'], 'C': ['9', 'Charlie']}) == {'Alice': 10, 'Bob': 10, 'Charlie': 9}
assert add_win_dict({'A': ['1', 'A'], 'B': ['2', 'B'], 'C': ['3', 'C'], 'D': ['4', 'D'], 'E': ['5', 'E']}) == {'E': 5, 'D': 4, 'C': 3}
assert add_win_dict({'Z': ['5', 'Z'], 'Y': ['4', 'Y'], 'X': ['3', 'X'], 'W': ['2', 'W'], 'V': ['1', 'V']}) == {'Z': 5, 'Y': 4, 'X': 3}
assert add_win_dict({'a': ['10', 'a'], 'b': ['5', 'b'], 'c': ['15', 'c'], 'd': ['15', 'd']}) == {'d': 15, 'c': 15, 'a': 10}
assert add_win_dict({'name1': ['7', 'player1'], 'name2': ['7', 'player2'], 'name3': ['7', 'player3']}) == {'player1': 7, 'player2': 7, 'player3': 7}

print("All test cases passed!")