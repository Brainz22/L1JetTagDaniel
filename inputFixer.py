import numpy as np

def add_ip(inputs):
    X = inputs

    ip = np.hypot(X[:,:, 9], X[:,:, 10])
    X_new = X[:, :, :9] # keep original upto 8th feature
    X_new = np.concatenate([X_new, ip[:, :, np.newaxis]], axis=-1) #add ip at 9th position
    X_new = np.concatenate([X_new, X[:, :, 11:14]], axis=-1) #append remaining features after 9th position

    # Sanity checks
    assert np.all(X_new[:,:, 8] == X[:,:, 8 ]) #original 8th feature unchanged
    assert np.all(X_new[:,:, 9] == ip) #ip appended correctly
    assert np.all(X_new[:,:, 10:] == X[:,:, 11:]) #Indices after 9th feature pushed back by 1

    return X_new


