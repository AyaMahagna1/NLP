import random

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim import models
from sys import argv
import regex as re

from jupyter_core.migrate import regex

# # Do the following only once!
#
# # Save the GloVe text file as a word2vec file for your use:
# glove2word2vec("C:\\Users\\aya19\\Desktop\\University\\ComputerScience\\NLP\\HW4\\glove.6B\\glove.6B.300d.txt",
# "kv_file.kv")
# # Load the file as KeyVectors:
# pre_trained_model = KeyedVectors.load_word2vec_format("kv_file.kv", binary=False)
# # Save the key vectors for your use:
# pre_trained_model.save("kv_file.kv")
#
# # Now, when handing the project, the KeyVector filename will be given as an argument. You may blacken
# # this part of the code again.
pattern = r'([\@\.\'\"\-\_\+\:\s0-9])+'


def pick_random_word(model):
    randomWord = random.choice(model.index_to_key)
    notWord = re.search(pattern, randomWord)
    while notWord:
        randomWord = random.choice(model.index_to_key)
        notWord = re.search(pattern, randomWord)
    # writing the secret word in file
    # with open(txt_file, 'w', encoding='utf-8') as file:
    #     file.write(randomWord)
    #     file.close()
    return randomWord


def start_game(model):
    print("\nWelcome to Semantle!, Let's start the game , but first read the instructions.\n\nInstructions:\n"
          "After each guess it will be printed the similarity of your guess to the correct word, so it may help you.\n"
          "You can choose the number of tries you want or unlimited number of guesses.\n"
          "If you fail to guess after 10 times we'll give you an extra hint about number of the characters.\n"
          "You can ask for a hint after every guess\n""You can exit the game and give up anytime.\n" "Enjoy and good "
          "luck!\n")
    game_on = True
    while game_on:
        secret_word = pick_random_word(model)
        count = 0
        limited = False
        hints = model.most_similar(positive=[secret_word])
        guesses = input('\nHi!, Please enter number of guesses, you can enter -1 for unlimited guesses: \n')
        if guesses != "-1":
            while not guesses.isdigit():
                guesses = input(
                    '\nInvalid input!, Please enter number of guesses, you can enter -1 for unlimited guesses: \n')
            if guesses.isdigit():
                guesses = int(guesses)
                limited = True
        while guesses != 0:
            if limited:
                if guesses == 0:
                    break
                else:
                    guesses = guesses - 1
            if count + 1 == 10:
                print(f'\nWe will give you an extra hint, the words consist of {len(secret_word)} characters.\n')
            word = input('\nGuess the word:\n')
            while word not in model:
                word = input("\nThe word you inserted is not in the vocabulary.Try a new guess:\n")
            if word == secret_word:
                print(f"\nYou're right! the word was {secret_word}, Congrats you won :)!\n")
                choice = input(
                    '\nIf you want to continue playing enter Y and if you want to exit the game enter N\n').lower()
                if choice == 'y':
                    break
                else:
                    if choice == 'n':
                        print('\nThank you for playing and goodbye!\n')
                        guesses = 0
                        game_on = False
                    else:
                        while choice != 'y' and choice != 'n':
                            choice = input(
                                '\nInvalid input!, if you want to continue playing enter Y and if you want to exit the '
                                'game enter N\n').lower()
                            if choice == 'y':
                                guesses = 0
                                break
                            else:
                                if choice == 'n':
                                    print('\nThank you for playing and goodbye!')
                                    game_on = False
                                    guesses = 0
                                    break
            else:
                print(
                    f'\nThe similarity is: {model.similarity(secret_word, word) :}')
                if guesses != 0:
                    choice = input(
                        '\nIf you want to continue playing enter Y and if you want to exit the game enter N\n').lower()
                    if choice == 'n':
                        print('\nThank you for playing and goodbye!\n')
                        game_on = False
                        guesses = 0
                    else:
                        while choice != 'y' and choice != 'n':
                            choice = input(
                                '\nInvalid input!, if you want to continue playing enter Y and if you want to exit the '
                                'game'
                                'enter N\n').lower()

                            if choice == 'n':
                                print('\nThank you for playing and goodbye!\n')
                                game_on = False
                                guesses = 0
                                break
                    if game_on and len(hints) > count:
                        choice = input(
                            '\nIf you want to get a hint enter Y and if you do not enter N\n').lower()
                        if choice == 'y':
                            print(f'\nHint: {hints[count][0]}, Similarity is: {hints[count][1]}\n')
                        else:
                            if choice != 'n':
                                while choice != 'y' and choice != 'n':
                                    choice = input(
                                        '\nInvalid input!, if you want to continue playing enter Y and if you want to '
                                        'exit'
                                        'the'
                                        'game enter N\n').lower()
                                if choice == 'y':
                                    print(
                                        f'\nHint: {hints[count][0]}, Similarity is: { hints[count][1]}\n')

                else:
                    print(
                        f'\nYou already reached the limited number of guesses!\nThe secret word was *** {secret_word} ***, '
                        f'good luck next time!\n')
                    choice = input(
                        'If you want to continue playing enter Y and if you want to exit the game enter N\n').lower()
                    if choice == 'n':
                        print('\nThank you for playing and goodbye!\n')
                        game_on = False
                        guesses = 0
                    while choice != 'y' and choice != 'n':
                        choice = input(
                            '\nInvalid input!, if you want to continue playing enter Y and if you want to exit the '
                            'game'
                            'enter N\n').lower()
                        if choice == 'n':
                            print(f'\nThe secret word was *** {secret_word} ***, Thank you for playing and goodbye!\n')
                            game_on = False
                            guesses = 0
                            break
            count += 1


if __name__ == "__main__":
    # kv_file = "kv_file.kv"
    # txt_file = 'word_file.txt'
    kv_file = argv[1]
    # You can load the vectors with the following (test with the one you saved!):
    pre_trained_model = KeyedVectors.load(kv_file, mmap='r')
    start_game(pre_trained_model)
