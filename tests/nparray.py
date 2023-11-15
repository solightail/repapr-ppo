import numpy as np

action_list = [-1, 0, 1]
tones = 6

action_arr = np.array([action_list] * tones)
empty = np.empty(tones)
flinf = float("inf")
npinf = np.inf

print(action_arr)
print(empty)
print(flinf)
print(npinf)