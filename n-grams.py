# Name: Aya Mahagna
# ID: 314774639
import math
import os
import re
from random import randint
from sys import argv
class Token:
    def __init__(self, tag, word):
        self.tag = tag
        self.word = word


class Sentence:
    def __init__(self, tokens, paragraph, length):
        self.tokens = tokens
        self.paragraph = paragraph
        self.length = length

    def append_token(self, token: Token):
        self.tokens.append(token)

    def __str__(self):
        return ' '.join([token.word for token in self.tokens])

    def __len__(self):
        return len(self.tokens)


class Corpus:
    dels = ["?", "!", ";", ":", "-", "'", '"', ")", "(", "’", "‘", ","]
    prefixes = ["U.S.", "U.A", "Mr.", "Mrs.", ".net", ".com", "W.E."]
    paragraph_del = "\n"
    sentence_del = "."
    token_del = " "
    title_del = "="

    def __init__(self):
        self.sentences = []

    def add_text_file_to_corpus(self, file_name):
        curr_file = open(file_name, "r", encoding="utf-8")
        content = curr_file.read()
        # count = 0
        # print("Sentences test:")
        for curr_paragraph in content.split(self.paragraph_del):
            if not self.check_empty(curr_paragraph):
                for sentence_id, curr_sentence in enumerate(self.split_to_sentences(curr_paragraph)):
                    if not self.check_empty(curr_sentence):
                        tokens = []
                        for curr_token in curr_sentence.split(self.token_del):
                            curr_token = curr_token.replace(self.title_del, "")
                            if not self.check_empty(curr_token):
                                tokens.append(Token('w', curr_token))
                        self.sentences.append(Sentence(tokens, 'p', len(tokens)))
                        # print(self.sentences[count])
                        # count += 1

    @staticmethod
    def check_empty(text: str):
        return re.compile("[\\n\\r]+").match(text) or text == ""

    def split_to_sentences(self, text):
        pattern = "(" + ''.join(
            map(re.escape, self.dels)) + ")"
        end = 0
        sentences_list = list()
        for content in re.finditer(r"\. ", text):
            curr_str = text[end: content.start() + 1]
            is_not_prefix = [not curr_str.endswith(prefix) for prefix in self.prefixes]
            if all(is_not_prefix):
                sentences_list.extend(re.split(pattern, curr_str))
                end = content.end()
        return sentences_list

    # Returns a random length of a sentence
    def get_random_length(self):
        random_index = randint(0, len(self.sentences) - 1)
        random_sen = self.sentences[random_index]
        total_len = len(random_sen.tokens)
        if random_sen.tokens[1].word == '<B>':
            total_len -= 1
        if random_sen.tokens[-1].word == '<E>':
            total_len -= 1
        return total_len


# NGram Model Class
# n=1 for Unigrams model
# n=2 for Bigrams model
# n=3 for Trigrams model
class NGramModel:
    def __init__(self, n, corpus):
        self.n = n
        self.corpus = corpus
        self.vocabulary = {}
        self.vocabulary_length = 0
        for corpus_sentence in corpus.sentences:
            for token in corpus_sentence.tokens:
                self.vocabulary_length += 1
                if token.word in self.vocabulary.keys():
                    self.vocabulary[token.word] += 1
                else:
                    self.vocabulary[token.word] = 1

    # Returns the probability of the sentence according laplace smoothing for Unigrams
    def probability_laplace_unigrams(self, corpus_sen):
        if self.n != 1:
            raise ValueError("Not Unigram!")
        total = 0
        words = corpus_sen.split(' ')
        last = words[len(words) - 1]
        if last[-1] == '.':
            words.pop(len(words) - 1)
            words.append(last[0: len(last) - 1])
        for word in words:
            if word in self.vocabulary.keys():
                current_probability = self.vocabulary[word] + 1
            else:
                current_probability = 1
            current_probability /= (self.vocabulary_length + len(self.vocabulary.keys()))
            total += math.log(current_probability)
        return pow(math.e, total)

    # Returns the probability of the sentence according laplace smoothing for Biagrams
    def probability_laplace_bigrams(self, corpus_sen):
        if self.n != 2:
            raise ValueError("Not Bigram!")
        words = corpus_sen.split(' ')
        last = words[len(words) - 1]
        if last[-1] == '.':
            words.pop(len(words) - 1)
            words.append(last[0: len(last) - 1])
        counter = 0
        for corpus_sen in self.corpus.sentences:
            if len(corpus_sen.tokens) > 0 and corpus_sen.tokens[0].word == words[0]:
                counter += 1
        result = math.log((1 + counter) / (len(self.corpus.sentences) + self.vocabulary_length))
        for words_index in range(len(words) - 1):
            pairs_counter = 0
            for corpus_sen in self.corpus.sentences:
                for j in range(len(corpus_sen.tokens) - 1):
                    if corpus_sen.tokens[j].word == words[words_index] and corpus_sen.tokens[j + 1].word == words[
                        words_index + 1]:
                        pairs_counter += 1
            current_probability = pairs_counter + 1
            if words[words_index] in self.vocabulary.keys():
                current_probability /= (self.vocabulary[words[words_index]] + self.vocabulary_length)
            else:
                current_probability /= self.vocabulary_length
            result += math.log(current_probability)
        counter = 0
        for corpus_sen in self.corpus.sentences:
            if len(corpus_sen.tokens) > 0 and corpus_sen.tokens[len(corpus_sen.tokens) - 1].word == words[-1]:
                counter += 1
        last_probability = (1 + counter) / (len(self.corpus.sentences) + self.vocabulary_length)
        result += math.log(last_probability)
        return pow(math.e, result)

    # Returns the probability of the sentence according to linear interpolation fot Trigrams
    def probability_in_linear_interpolation_trigram(self, corpus_sen: str):
        if self.n != 3:
            raise ValueError("Not Trigram!")
        lambda1, lambda2 = 0.8, 0.15
        lambda3 = 1 - lambda1 - lambda2
        words = corpus_sen.split(' ')
        last = words[len(words) - 1]
        if last[-1] == '.':
            words.pop(len(words) - 1)
            words.append(last[0: len(last) - 1])  # removing the end point
        if words[0] in self.vocabulary.keys():
            result = math.log(self.vocabulary[words[0]] / self.vocabulary_length)
        else:
            result = 0
        if len(words) == 1:
            if result == 0:
                return 0
            return pow(math.e, result)
        val1, val2 = 0.65, 0.35
        counter = 0
        for corpus_sen in self.corpus.sentences:
            for words_index in range(len(corpus_sen.tokens) - 1):
                if corpus_sen.tokens[words_index].word == words[0] and corpus_sen.tokens[words_index + 1].word == words[
                    1]:
                    counter += 1
        if words[0] in self.vocabulary.keys():
            if counter != 0:
                result += val1 * math.log(counter / self.vocabulary[words[0]])
        if words[1] in self.vocabulary.keys():
            result += val2 * math.log(self.vocabulary[words[1]] / self.vocabulary_length)
        if len(words) == 2:
            return pow(math.e, result)
        for k in range(2, len(words)):
            if words[k] in self.vocabulary.keys():
                current_probability = lambda3 * math.log(self.vocabulary[words[k]] / self.vocabulary_length)
            else:
                current_probability = 0
            counter = 0
            for corpus_sen in self.corpus.sentences:
                for _l in range(len(corpus_sen.tokens) - 1):
                    if corpus_sen.tokens[_l].word == words[k - 1] and corpus_sen.tokens[_l + 1].word == words[k]:
                        counter += 1
            if words[k - 1] in self.vocabulary.keys() and self.vocabulary[words[k - 1]] != 0 and \
                    counter != 0:
                current_probability += lambda2 * math.log(counter / self.vocabulary[words[k - 1]])
            triples = 0
            pairs = 0
            for corpus_sen in self.corpus.sentences:
                for j in range(len(corpus_sen.tokens) - 2):
                    if corpus_sen.tokens[j].word == words[k - 2] and corpus_sen.tokens[j + 1].word == words[k - 1]:
                        pairs += 1
                        if corpus_sen.tokens[j + 2].word == words[k]:
                            triples += 1
            if pairs != 0 and triples != 0:
                current_probability += lambda1 * math.log(triples / pairs)
            result += current_probability
        return pow(math.e, result)

    # Returns a random sentence in Unigrams model
    def random_sentence_unigram(self, sentence_length):
        if self.n != 1:
            raise ValueError("Not Unigram")
        start_sen = ['<B>']
        end_sen = '<E>'
        possibilities = []
        for corpus_sen in self.corpus.sentences:
            for token in corpus_sen.tokens:
                if token.word != '<B>':
                    possibilities.append(token.word)
        for k in range(sentence_length):
            random_index = randint(0, len(possibilities) - 1)  # using random.randint to get a random index
            start_sen.append(possibilities[random_index])
            if start_sen[-1] == end_sen:
                return start_sen
        if start_sen[-1] != end_sen:
            start_sen.append(end_sen)
        return start_sen

    # Returns a random sentence in Bigrams model
    def random_sentence_bigram(self, sentence_length):
        if self.n != 2:
            raise ValueError("Not Bigram")
        start_sen = ['<B>']
        end_sen = '<E>'
        for k in range(sentence_length):
            possibilities = []
            for corpus_sen in self.corpus.sentences:
                for j in range(len(corpus_sen.tokens) - 1):
                    if corpus_sen.tokens[j].word == start_sen[-1]:
                        possibilities.append(corpus_sen.tokens[j + 1].word)
            random_index = randint(0, len(possibilities) - 1)
            start_sen.append(possibilities[random_index])
            if start_sen[-1] == end_sen:
                return start_sen
        if start_sen[-1] != end_sen:
            start_sen.append(end_sen)
        return start_sen

    # Returns a random sentence in Trigrams model
    def random_sentence_trigram(self, sentence_length):
        if self.n != 3:
            raise ValueError("Not Trigram")
        start_sen = ['<B>']
        end_sen = '<E>'
        possibilities = []
        for corpus_sen in self.corpus.sentences:
            possibilities.append(corpus_sen.tokens[1].word)
        random_index = randint(0, len(possibilities) - 1)
        start_sen.append(possibilities[random_index])
        for k in range(sentence_length - 1):  # already added a token
            possibilities = []
            for current_sen in self.corpus.sentences:
                for j in range(len(current_sen.tokens) - 2):
                    if current_sen.tokens[j].word == start_sen[-2] and current_sen.tokens[j + 1].word == start_sen[-1]:
                        possibilities.append(current_sen.tokens[j + 2].word)
            random_index = randint(0, len(possibilities) - 1)
            start_sen.append(possibilities[random_index])
            if start_sen[-1] == end_sen:
                return start_sen
        if start_sen[-1] != end_sen:
            start_sen.append(end_sen)
        return start_sen

    # Returns a random sentence according to n
    def random_sentence(self, sentence_length):
        if sentence_length == 0:
            return ''
        self.vocabulary['<E>'] = len(self.corpus.sentences)
        if self.n == 1:
            return self.random_sentence_unigram(sentence_length)
        if self.n == 2:
            return self.random_sentence_bigram(sentence_length)
        if self.n == 3:
            return self.random_sentence_trigram(sentence_length)
        return ''


if __name__ == '__main__':
    corpus_dir = "../HW3/text_files"
    output_file = "output.txt"
    # corpus_dir = argv[1]
    # output_file = argv[2]
    output_text = ""
    corpus1 = Corpus()
    for wiki_file in os.listdir(corpus_dir):
        if wiki_file.endswith(".txt"):
            corpus1.add_text_file_to_corpus(os.path.join(corpus_dir, wiki_file))
    unigrams = NGramModel(1, corpus1)  # 1 for Unigrams
    bigrams = NGramModel(2, corpus1)  # 2 for Bigrams
    trigrams = NGramModel(3, corpus1)  # 3 for Trigrams

    # Task 1: Probabilities for the given sentences
    sentences = [
        "May the Force be with you.",
        "I’m going to make him an offer he can’t refuse.",
        "Ogres are like onions.",
        "You’re tearing me apart, Lisa!",
        "I live my life one quarter at a time."
    ]
    output_text += "*** Sentence Predictions ***\n\n"
    for ngram in (unigrams, bigrams, trigrams):
        if ngram.n == 1:
            output_text += "Unigrams Model:\n\n"
        elif ngram.n == 2:
            output_text += "Bigrams Model:\n\n"
        else:
            output_text += "Trigrams Model:\n\n"
        for sentence in sentences:
            output_text += sentence + "\n"
            output_text += "Probability: "
            if ngram.n == 1:
                output_text += str(math.log(ngram.probability_laplace_unigrams(corpus_sen=sentence)))
            elif ngram.n == 2:
                output_text += str(math.log(ngram.probability_laplace_bigrams(corpus_sen=sentence)))
            else:
                output_text += str(math.log(ngram.probability_in_linear_interpolation_trigram(corpus_sen=sentence)))
            output_text += '\n\n'

    # Task 2
    for sentence in corpus1.sentences:
        new_sen = [Token('w', '<B>')]
        new_sen.extend(sentence.tokens)
        new_sen.append(Token('w', '<E>'))
        sentence.tokens = new_sen
    output_text += "*** Random Sentence Generation ***\n\n"
    for ngram in (unigrams, bigrams, trigrams):
        if ngram.n == 1:
            output_text += "Unigrams Model:\n\n"
        elif ngram.n == 2:
            output_text += "\nBigrams Model:\n\n"
        else:
            output_text += "\nTrigrams Model:\n\n"
        for i in range(5):
            random_length = ngram.corpus.get_random_length()
            new_sentence = ngram.random_sentence(sentence_length=random_length)
            output_text += '<'
            for index in range(1, len(new_sentence) - 1):
                output_text += new_sentence[index]
                output_text += ' '
            output_text += '>'
            output_text += '\n'
    # Write the output of task 1 and 2 to the file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    f.close()
    print(output_text)
