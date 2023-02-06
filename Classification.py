# Name: Aya Mahagna
# ID: 314774639

from sys import argv
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def down_sample(output, csv_data):
    review = csv_data.iloc[:, 0].values
    recommended = csv_data.iloc[:, 1].values
    sub_sampled_class = np.bincount(recommended).argmax()
    majority_indexes = np.where(recommended == sub_sampled_class)[0]
    minority_class = np.where(recommended != sub_sampled_class)[0]
    output.write("Before Down-sampling:\n")
    text = "Recommended: "
    text += str(len(majority_indexes))
    text += "\n"
    text += "Not Recommended: "
    text += str(len(minority_class))
    text += "\n\n"
    output.write(text)
    down_sampled_indexes = np.random.choice(majority_indexes,
                                            size=len(minority_class), replace=False)
    chosen = np.concatenate((down_sampled_indexes, np.where(recommended != sub_sampled_class)[0]))
    np.random.shuffle(chosen)
    review = review[chosen]
    recommended = recommended[chosen]
    majority_indexes = np.where(recommended == sub_sampled_class)[0]
    minority_class = np.where(recommended != sub_sampled_class)[0]
    output.write("After Down-sampling:\n")
    text = "Recommended: "
    text += str(len(majority_indexes))
    text += "\n"
    text += "Not Recommended: "
    text += str(len(minority_class))
    text += "\n\n"
    output.write(text)
    return review, recommended, csv_data.iloc[chosen, :]


def tokenize(text, token=spacy.blank('en').tokenizer):
    return token(text)


def custom_feature_vector(data):
    vector = np.zeros((len(data), 2))
    data['Review Text'] = data['Review Text'].apply(tokenize)
    data['Review Text'] = data['Review Text'].apply(lambda vec: [token for token in vec if token.text != ''])
    positive = ["wonderful", "silky", "sexy", "comfortable", "love", "pretty", "glad",
                "nicely", "fun", "flirty", "fabulous", "compliments", "great", "flattering", "well",
                "perfect", "gorgeous", "perfectly", "feminine", "nice", "style", "perfection", "happy", "cute",
                "good", "cozy", "stylish", "classic", "beautifully", "super", "lovely", "unique", "roomy", "better",
                "adorable", "soft", "practical", "loved"]
    negative = ["small", "but", "disappointing", "tight", "cheap", "disappointed", "wouldn't", "awkward", "horrible",
                "didn't "
        , "replace", "annoying", "not", "returned", "returning", "poor", "terrible", "gaps", "sadly", "cheeky",
                "unflattering", "cut", "unfortunately", "scratchy", "odd", "short", "squat", "uncomfortable",
                "stiff", "boxy", "don't", "itchy", "torn", "offensive", "isn't"]
    vector[:, 0] = data['Review Text'].apply(
        lambda pos: len([token for token in pos if token.text.lower() in positive]))
    vector[:, 1] = data['Review Text'].apply(
        lambda neg: len([token for token in neg if token.text.lower() in negative]))
    return StandardScaler().fit_transform(vector)


if __name__ == "__main__":
    csv_file = argv[1]  # The path to the csv file as downloaded from kaggle after unzipping
    output_file = argv[2]  # The path to the text file in which the output is written onto
    file = open(output_file, "w")
    # Task 1 - Down-Sampling #
    fields = ['Review Text', 'Recommended IND']
    data = pd.read_csv(csv_file, usecols=fields)
    data = data.dropna()
    # Tokenization using spacy
    token = spacy.blank('en').tokenizer
    data['Review Text'] = data.apply(lambda row: token(row['Review Text']), axis=1)
    data['Review Text'] = data['Review Text'].apply(lambda ind: [token for token in ind if not token.is_space])
    data['Review Text'] = data['Review Text'].apply(lambda ind: [token for token in ind if token.text != ''])
    data['Review Text'] = data['Review Text'].apply(lambda ind: ' '.join([token.text for token in ind]))
    review_text, recommended_ind, data = down_sample(file, data)
    # BoW classification
    vectors = TfidfVectorizer()
    review_text = vectors.fit_transform(review_text)
    x_train, x_test, y_train, y_test = train_test_split(review_text, recommended_ind, test_size=0.3, random_state=0)
    knn = KNeighborsClassifier(50)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    file.write("== BoW Classification ==\n")
    file.write("Cross Validation Accuracy: %.3f\n" % roc_auc_score(y_test, y_pred))
    file.write(classification_report(y_test, y_pred))
    # 10-fold cross validation
    file.write("\n== Custom Feature Vector Classification ==\n")
    vec = custom_feature_vector(data)
    knn = KNeighborsClassifier(20)
    x_train, x_test, y_train, y_test = train_test_split(vec, recommended_ind, test_size=0.3, random_state=0)
    knn.fit(x_train, y_train)
    vectors = cross_val_score(knn, vec, recommended_ind, cv=10)
    file.write("Cross Validation Accuracy: %.3f \n" % vectors.mean())
    file.write(classification_report(y_test, knn.predict(x_test)))
    file.close()
