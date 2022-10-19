from data_generator import get_data
from likelihood import Intensity_Model
import numpy as np


if __name__ == "__main__":
    n_ind = 10
    n_times = 40
    Times, X, Cens = get_data(n_ind=n_ind, n_times=n_times)

    Y = [[0 for _ in range(len(X[0][0]))] for _ in range(len(X))]

    intensity_model = Intensity_Model(X, Times, Cens, Y)
    print(np.array(intensity_model.Y).shape)
    intensity_model.f_likelihood([2, 1, 1, 0.5], "beta", intensity_model.Y)