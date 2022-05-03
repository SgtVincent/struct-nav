""" 
Note 1: There are 31 classes (including 'no label') for regions in Matterport 3D 
ref: https://github.com/niessner/Matterport/blob/master/data_organization.md

Note 2: here are 41 classes (including 'unlabeled') for objects in Matterport 3D 
ref: https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
"""


class Matterport3DConfig:
    def __init__(self):

        number_region_class = 31
        number_object_class = 41
