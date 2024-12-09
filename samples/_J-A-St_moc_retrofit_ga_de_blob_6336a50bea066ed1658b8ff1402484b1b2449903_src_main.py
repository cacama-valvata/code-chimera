import os
import sys
import code
import pickle

# if sys.platform == "linux":
#     import resource
#     resource.setrlimit(resource.RLIMIT_DATA, (2147483648,2147483648)) # resource.RLIMIT_AS ; pthread_create failed: https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing

from read_data.read_case_study_data import CaseStudy
from read_data.read_algorithm_parameter import AlgorithmParameter
from algorithm.genetic_algorithm import GeneticAlgorithm

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')
    case_study = CaseStudy('Zweifel.xlsx')
    algorithm_parameter = AlgorithmParameter('AlgorithmParameter_short_comp.xlsx')
    genetic_algorithm = GeneticAlgorithm(case_study, algorithm_parameter)
    hall_of_fame = genetic_algorithm.genetic_algorithm()
    file = open("HallOfFame.pkl", "wb")
    pickle.dump([[[hall_of_fame[z][y][x] for x in range(len(hall_of_fame[z][y]))] for y in range(len(hall_of_fame[z]))] for z in range(len(hall_of_fame.items))], file)
    file.close()
    print(210*"-")
    print("\nMulti-objective optimization is finished. Do you want to access the results? Input yes/no?\n")
    print(210*"-")
    answer = input()
    while answer != 'yes' and answer != 'no':
        print(210*"-")
        print("\nNon-valid input. Input ''yes,, to access the results or ''no,, to terminate\n")
        print(210*"-")
        answer = input()
    if answer == 'no':
        sys.exit(0)
    elif answer == 'yes':
        print(20*"-")
        print("\nPress exit() to exit\n")
        print(20*"-")
        code.interact(local=locals())


if __name__ == "__main__":
    main()
