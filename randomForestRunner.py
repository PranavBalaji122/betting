from randomForestRegPredictor import run
from consienstyTest import getConsistency


def runner():
    player_names = getConsistency('trb')
    player_names.remove('Jordan Walsh')
    player_names.remove('Darius Bazley')
    player_names = ['Bam Adebayo']
    # print(player_names)
    opponents = [
     "OKC" ]
    output = {}
    print(run(player_names[0],opponents[0],'trb'))
    # for i in range(len(opponents)):
    #     output[player_names[i]] = run(player_names[i],opponents[i],'pts')
        
    # print(output)


if __name__ == '__main__':
    runner()
