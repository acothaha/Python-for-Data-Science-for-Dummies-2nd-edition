# function of 2 alternative text encoding strategies
def comparison_test(text):
    import sklearn.feature_extraction.text as txt
    h_trick = txt.HashingVectorizer(n_features=20, binary=True,
                                    norm=None)
    oh_encoder = txt.CountVectorizer()
    oh_encoded = oh_encoder.fit_transform(text)
    hashing = h_trick.transform(text)
    return oh_encoded, hashing

