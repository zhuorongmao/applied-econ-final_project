
from function_library import GeneratePermutations, generate_random_numbers

if len(GeneratePermutations(4)==24):
    print('GeneratePermutations works')

if sum(generate_random_numbers(3,0.5)==1-0.5):
    print('generate_random_numbers works')

