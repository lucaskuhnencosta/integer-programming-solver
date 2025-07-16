from typing import List
import numpy as np

class ModelCanonicalizer:
    def __init__(self):
        self.name="ModelCanonicalizer"

    def apply(self,instance):
        A=instance.A
        b=instance.b
        sense=instance.sense
        row_names=instance.row_names

        for i in range(len(sense)):
            if sense[i]=='G':
                A[i,:] *=-1
                b[i]*=-1
                sense[i]='L'