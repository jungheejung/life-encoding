#!/usr/bin/env python
"""
1. select vertex of interest (extract vertex number from suma)
2. use pymvpa to extract vertices within searchlight radius of 5mm
"""

from mvpa2.datasets.gifti import surf_from_any

roi_dict = {"ips_rh": 39287,
"loc_rh": 26304,
"vt_rh": 40420
}
pial = surf_from_any('/Users/h/suma-fsaverage6/rh.pial.gii')

for ind in range(len(roi_dict)):
    neighbs = pial.circlearound_n2d(roi_dict.values()[ind], 5)
    fname = roi_dict.keys()[ind] + '.txt'
    textfile = open(fname, "w")

    for element in neighbs.keys():
        textfile.write(str(element) + "\n")
    textfile.close()
