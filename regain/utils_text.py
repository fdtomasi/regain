"""Utils to work with text data.

TODO clean tree structure for this utils.
"""
import numpy as np


def display_topics(
        H, W, feature_names, documents, n_top_words, n_top_documents,
        print_docs=True):
    topics = []
    for topic_idx, topic in enumerate(H):
        topics.append(
            " ".join(
                [
                    feature_names[i] + " (%.3f)" % topic[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]
                ]))

        print("Topic %d: %s" % (topic_idx, topics[-1]))
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][:n_top_documents]
        if print_docs:
            for i, doc_index in enumerate(top_doc_indices):
                print("doc %d: %s" % (doc_index, documents[doc_index]))
    return topics


def logentropy_normalize(X):
    P = X / X.values.sum(axis=0, keepdims=True)
    E = 1 + (P * np.log(P)).fillna(0).values.sum(
        axis=0, keepdims=True) / np.log(X.shape[0] + 1)
    return E * np.log(1 + X)
