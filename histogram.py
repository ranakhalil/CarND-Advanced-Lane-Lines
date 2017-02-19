#!/usr/bin/python
import sys
import random
import matplotlib.pyplot as plt
sampleSize = 50  ## Try making this value bigger
random.seed()
## Let's start with the simplest thing and collect a sample
## of real-number random variates between 0 and 1
realRandomVariates = []
for i in range(sampleSize):
    newValue = random.random()
    realRandomVariates.append(newValue)
plt.hist(realRandomVariates,10)
plt.xlabel('Number range')
plt.ylabel('Count')
plt.show()