#  Packing relevant (see variable idxsPose and idxsHand) keypoints stored as json files into a single h5 file. 
#  Format: x, y, likelihood, x, y, likelihood, ....
#  x == 0, y == 0 means missing keypoint 

import sys
import math
import re
import os
import json
import h5py


import numpy
from walkDir import walkDir


def findTheTallest(people):
  Lmax = None
  theTallest = None
  for dude in people:
    x0 = dude[3 * 2 + 0]
    y0 = dude[3 * 2 + 1]
    x1 = dude[3 * 5 + 0]
    y1 = dude[3 * 5 + 1]
    L = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)
    if Lmax is None or Lmax < L:
      Lmax = L
      theTallest = dude
  return theTallest


def selectPoints(points, keepThis):
  points2 = []
  for i in keepThis:
    points2.append(points[3 * i + 0])
    points2.append(points[3 * i + 1])
    points2.append(points[3 * i + 2])
  return points2


def noNones(l):
  l2 = []
  for i in l:
    if not i is None:
      l2.append(i)
  return l2


def loadData(dname):
  fnames = walkDir(dname = dname, filt = r"\.json$")
  fnames.sort()
  frames = []
  for fname in fnames:
    p = re.search(r".*_(\d+)_keypoints\.json$", fname)
    
    with open(fname) as json_data:
      data = json.load(json_data)
    if len(data["people"]) == 0:
      continue
      
    i = int(p.group(1))
    while len(frames) < i + 1:
      frames.append(None)

    idxsPose = [0, 1, 2, 3, 4, 5, 6, 7]
    idxsHand = range(21)    

    people = []
    for dude in data["people"]:
      pointsP = dude["pose_keypoints_2d"]
      pointsLH = dude["hand_left_keypoints_2d"]
      pointsRH = dude["hand_right_keypoints_2d"]
      pointsP = selectPoints(pointsP, idxsPose)
      pointsLH = selectPoints(pointsLH, idxsHand)
      pointsRH = selectPoints(pointsRH, idxsHand)
      points = pointsP + pointsLH + pointsRH
      people.append(points)

    points = findTheTallest(people)

    if points is None:
      points = 3 * (len(idxsPose) + 2 * len(idxsHand)) * [0.0]

    if not points[0] == 0.0:
      frames[i] = points
    
  return numpy.asarray(noNones(frames), dtype="float32")


if __name__ == "__main__":

  #dnameIn = "train_json2"
  #fnameOut = "train_json2_keypoints-raw.h5"
  dnameIn = sys.argv[1] #cesta k "../../data/test_json"
  fnameOut = sys.argv[2] #cesta a h5 "../../data/test_json_keypoints-raw.h5"
  
  recs = {}
  for fname in walkDir(dnameIn, filt=r"\.[jJ][sS][oO][nN]$"):
    dname = re.sub(r"(.*)[/\\].*", r"\1", fname)
    key = re.sub(r"[^\\/]*[/\\]", "", dname)
    key = re.sub(r"_openpose/[a-z]+_", "", key)
    recs[key] = dname
    
  hf = h5py.File(fnameOut, "w")
  for key in recs:
    print(key)
    data = loadData(recs[key])
    print(data.shape)
    hf.create_dataset(key, data=data, dtype="float32")    
  hf.close()
