"""Independent synthetic checks of the statistic's edge cases; no human data."""
import numpy as np
from scipy.stats import spearmanr
from reviewer_diagnostics import correlations,row_mean

a=np.array([[1,2,2,4,5],[1,2,np.nan,4,5],[3,3,3,3,3],[np.nan]*5,[1,2,np.nan,np.nan,np.nan]],float)
b=np.array([[5,2,2,1,4],[2,np.nan,3,5,4],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],float)
actual=correlations(a,b)
for i in [0,1]:
    mask=np.isfinite(a[i])&np.isfinite(b[i])
    assert abs(actual[i]-spearmanr(a[i,mask],b[i,mask]).statistic)<1e-12
assert np.isnan(actual[2:]).all()
assert np.isnan(row_mean(a)[3])
print('Passed: ties, pairwise missingness, constant profiles, empty profiles and fewer than three common items.')
