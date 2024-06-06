import numpy as np
from cation_electronegativity import A_site_cation_electronegativity, B_site_cation_electronegativity
import pandas as pd
from itertools import product

for key in B_site_cation_electronegativity.keys():
    if key in ["Co","Fe"]:
        valence = []
        possible_cation = list(B_site_cation_electronegativity[key].keys())
        possible_ratio = np.arange(0.0,1.01,0.01)
        possible_ratio = np.append(
            possible_ratio,
            [
                0.294,0.706,
                0.489,0.511,
                0.247,0.753,
                0.0793,0.3018,0.6189,
                0.1123,0.3204,0.5673,
                0.1252,0.3097,0.5651,
                0.1541,0.8459,
                0.2535,0.7465,
            ]
        )
        all_ratio_comb = list(product(possible_ratio, repeat=len(possible_cation)))
        possible_ratio_comb = []
        for comb in all_ratio_comb:
            if np.sum(comb) == 1:
                valence.append([np.sum(np.multiply(comb,possible_cation))])
                possible_ratio_comb.append(comb)
        df = pd.DataFrame(np.concatenate((np.asarray(possible_ratio_comb),valence),axis=1), columns=possible_cation+["valence"])
        df.to_excel("oxi_states_%s.xlsx"%(key),index=False)