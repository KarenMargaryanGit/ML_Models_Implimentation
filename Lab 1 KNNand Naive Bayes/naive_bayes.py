class MyNaiveBayes:
    def __init__(self, smoothing=False):
        # initialize Laplace smoothing parameter
        self.smoothing = smoothing

    def fit(self, X_train, y_train):
        # use this method to learn the model
        # if you feel it is easier to calculate priors
        # and likelihoods at the same time
        # then feel free to change this method
        self.X_train = X_train
        self.y_train = y_train
        self.priors = self.calculate_priors()
        self.likelihoods = self.calculate_likelihoods()

    def predict(self, X_test):
        # recall: posterior is P(label_i|feature_j)
        # hint: Posterior probability is a matrix of size
        #       m*n (m samples and n labels)
        #       our prediction for each instance in data is the class that
        #       has the highest posterior probability.
        #       You do not need to normalize your posterior,
        #       meaning that for classification, prior and likelihood are enough
        #       and there is no need to divide by evidence. Think why!
        # return: a list of class labels (predicted)
        ##### YOUR CODE STARTS HERE #####

        # print(X_test)

        prediction = []

        for ind in X_test.index:
            res = []
            for ans in self.priors.keys():
                k = 1
                for columns in X_test.columns:
                    # print(X_test[columns][ind])
                    # print(self.likelihoods[f"{ans},{columns},{X_test[columns][ind]}"])
                    k *= self.likelihoods[f"{ans},{columns},{X_test[columns][ind]}"]
                k *= self.priors[ans]
                # print(self.priors[[ans]])
                res.append(k)

            # print(self.priors.keys()[res.index(max(res))])
            prediction.append(self.priors.keys()[res.index(max(res))])

        ##### YOUR CODE ENDS HERE #####
        return prediction

    def calculate_priors(self):
        # recall: prior is P(label=l_i)
        # hint: store priors in a pandas Series or a list
        ##### YOUR CODE STARTS HERE #####

        # print(self.y_train.value_counts(normalize=True))
        priors = self.y_train.value_counts(normalize=True)

        # priors = self.y_train.value_counts()
        
        # print(type(priors))

        # priors += 1 * int(self.smoothing)
        # print(priors)

        # priors /= sum(priors) + len(priors) * int(self.smoothing)
        
        # print(priors)

        ##### YOUR CODE ENDS HERE #####
        return priors

    def calculate_likelihoods(self):
        # recall: likelihood is P(feature=f_j|label=l_i)
        # hint: store likelihoods in a data structure like dictionary:
        #        feature_j = [likelihood_k]
        #        likelihoods = {label_i: [feature_j]}
        #       Where j implies iteration over features, and
        #             k implies iteration over different values of feature j.
        #       Also, i implies iteration over different values of label.
        #       Likelihoods, is then a dictionary that maps different label
        #       values to its corresponding likelihoods with respect to feature
        #       values (list of lists).
        #
        #       NB: The above pseudocode is for the purpose of understanding
        #           the logic, but it could also be implemented as it is.
        #           You are free to use any other data structure
        #           or way that is convenient to you!
        #
        #       More Coding Hints: You are encouraged to use Pandas as much as
        #       possible for all these parts as it comes with flexible and
        #       convenient indexing features which makes the task easier.
        ##### YOUR CODE STARTS HERE #####

        likelihoods = dict()
        for i in self.priors.keys():
            # print(i)
            # print(self.X_train[self.y_train == i])

            # print(self.X_train.columns)

            for j in self.X_train.columns:
                # print(i,j,  self.X_train[self.y_train == i][j].value_counts(normalize = True))

                # data = self.X_train[self.y_train == i][j].value_counts(normalize=True)

                data = self.X_train[self.y_train == i][j].value_counts()

                # print(type(data))
                data += 1 * int(self.smoothing)
                # print(len(self.priors))
                data /= sum(data)
                for k in data.keys():
                    likelihoods[f"{i},{j},{k}"] = data[k]

        # print(likelihoods)

        ##### YOUR CODE ENDS HERE #####
        return likelihoods