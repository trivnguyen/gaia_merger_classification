import numpy as np

StarIndicies = np.arange(598334698)
np.random.shuffle(StarIndicies)

with open('data/StarIndiciesTest.txt', 'w') as f:
    for i, star in enumerate(StarIndicies):
        f.write('{0},{1}\n'.format(i, star))

np.save('data/StarIndexList.npy', StarIndicies)
