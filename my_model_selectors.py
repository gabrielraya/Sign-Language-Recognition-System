import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        
            BIC = −2 log L + p log N
            L = log_likelihood
            p = number of parameters in model
            N = number of data points
            
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        # The lower the BIC value the better the model
        lowest_bic = float("inf")
        best_model = None

        for hidden_states_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(hidden_states_number)
                # defines the L = log_likelihood parameter
                log_likelihood = model.score(self.X, self.lengths)
                # defines the p = number of parameters in model to be  p = n^2 + 2*d*n - 1
                parameters_number = (hidden_states_number * hidden_states_number + 
                                     2 * model.n_features * hidden_states_number - 1)
                # BIC = −2 log L + p log N
                bic_score = -2 * log_likelihood + parameters_number * np.log(len(self.X))
                
                if bic_score < lowest_bic:
                    lowest_bic = bic_score
                    best_model = model
            # pylint: disable=broad-except
            # exceptions vary and occurs deep in other external classes
            except Exception:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float("-inf")
        best_model = None
        number_of_words = len(self.hwords.keys())

        for hidden_states_number in range(self.min_n_components, self.max_n_components):
            sum_log_likelihood_other_words = 0
            try:
                # train the model 
                model = GaussianHMM(n_components=hidden_states_number, n_iter=1000).fit(self.X, self.lengths)
                sum_log_likelihood_matching_words = model.score(self.X, self.lengths)

                for word in self.hwords.keys():
                    other_word_data_points, lengths = self.hwords[word]
                    log_likelihood_other_words = model.score(other_word_data_points, lengths)
                    sum_log_likelihood_other_words += log_likelihood_other_words
            # pylint: disable=broad-except
            # exceptions vary and occurs deep in other external classes       
            except Exception as e:
                break
            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            dic_score = sum_log_likelihood_matching_words - (1 / (number_of_words - 1)) * (
                sum_log_likelihood_other_words - sum_log_likelihood_matching_words)

            if dic_score > best_score:
                best_score = dic_score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    ''' Helper function
        returns the log_likelihood value given a base_model
    '''
    def run_model(self, n_components, X_train, lengths_train, X_test, lengths_test):
        model = GaussianHMM(n_components=n_components, covariance_type="diag",n_iter=1000, 
                            random_state=self.random_state,verbose=False).fit(X_train, lengths_train)
        log_likelihood = model.score(X_test, lengths_test)
        return log_likelihood

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # hide hmmlearn==0.2.0 warnings

        best_model = None
        highest_average = float('-inf')
        n_splits = min(len(self.sequences), 3)

        for n_components in range(self.min_n_components,self.max_n_components + 1):
            total_logL = 0
            iterations = 0
            try:
                if n_splits > 1:
                    split_method = KFold(n_splits=n_splits)
                    seq_splits = split_method.split(self.sequences)
                    for cv_train_idx, cv_test_idx in seq_splits:
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        # get log_likelihood after running model
                        log_likelihood = self.run_model(n_components, X_train,lengths_train, X_test,lengths_test)
                        total_logL += log_likelihood
                        iterations += 1
                else:
                    log_likelihood = self.run_model(n_components, self.X, self.lengths,self.X, self.lengths)
                    total_logL += log_likelihood
                    iterations += 1
                average_logL = total_logL / iterations
                if average_logL > highest_average:
                    highest_average = average_logL
                    best_model = self.base_model(n_components)

            # pylint: disable=broad-except
            # exceptions vary and occurs deep in other external classes
            except Exception:
                continue

        return best_model