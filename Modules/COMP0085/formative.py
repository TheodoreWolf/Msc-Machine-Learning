import numpy as np

cov = 1/6 * np.array([[7,-2,-2,1],
                        [-2,7,1,-2],
                        [-2,1,7,-2],
                        [1,-2,-2,7]])
mu = np.array([0,1,1,0])

print(mu.T @ np.linalg.inv(cov) @ mu)