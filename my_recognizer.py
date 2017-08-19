import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # Implement the recognizer return probabilities, guesses
    for word_index, _ in test_set.get_all_Xlengths().items():
        word_data_points, length = test_set.get_item_Xlengths(word_index)

        log_likelihood_dict = {}
        for word, model in models.items():
            try:
                log_likelihood = model.score(word_data_points, length)
                log_likelihood_dict[word] = log_likelihood
            except Exception as e:
                log_likelihood_dict[word] = float("-inf")

        probabilities.append(log_likelihood_dict)
        guesses.append(max(log_likelihood_dict, key=log_likelihood_dict.get))

    return probabilities, guesses
