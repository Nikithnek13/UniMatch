import numpy as np

def idx(vals, lst):
    return np.array([lst.index(v) if v in lst else -1 for v in vals])

def features(s, med, caste, gender, region, bcode, fee, strength, enc):
    """Feature matrix shared by training and prediction."""
    s = np.asarray(s, float); med = np.asarray(med, float)
    return np.column_stack([np.log(s / med), np.log(s),
        idx(caste, enc["caste"]), idx(gender, enc["gender"]), idx(region, enc["region"]),
        idx(bcode, enc["branch"]), fee, strength])
