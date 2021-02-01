# ChaLearn-SLR
These codes generate the results for 2021 Looking at People Large Scale Signer Independent Isolated SLR CVPR Challenge.

http://chalearnlap.cvc.uab.es/challenge/43/description/

## OpenPose joint indexes

The locations are from the person's point of view (as caputured in video, must not reflect reality, if the video was mirrored).
E.g. Left shoulder means the 'left' shoulder of the person which is on the right side of the image.
For hand joints and parts refer to https://www.mdpi.com/sensors/sensors-20-01074/article_deploy/html/images/sensors-20-01074-g001.png


Index | Body part | Index | Body part | Index | Body part
--- | --- | --- | --- | --- | ---
0 | head | 8 | left hand wrist | 29 | right hand wrist
1 | neck | 9 | left root of thumb (CMC) | 30 | right root of thumb (CMC)
2 | right shoulder | 10 | left thumb 1st joint (MCP) | 31 | right thumb 1st joint (MCP)
3 | right elbow | 11 | left thumb 2nd joint (DIP) | 32 | right thumb 2nd joint (DIP)
4 | right wrist | 12 | left thumb fingertip | 33 | right thumb fingertip
5 | left shoulder | 13 | left index finger MCP | 34 | right index finger MCP
6 | left elbow | 14 | left index finger PIP | 35 | right index finger PIP
7 | left wrist | 15 | left index finger DIP | 36 | right index finger DIP
 | | | 16 | left index fingertip | 37 | right index fingertip
 | | | 17 | left middle finger MCP | 38 | right middle finger MCP
 | | | 18 | left middle finger PIP | 39 | right middle finger PIP
 | | | 19 | left middle finger DIP | 40 | right middle finger DIP
 | | | 20 | left middle fingertip | 41 | right middle fingertip
 | | | 21 | left ring finger MCP | 42 | right ring finger MCP
 | | | 22 | left ring finger PIP | 43 | right ring finger PIP
 | | | 23 | left ring finger DIP | 44 | right ring finger DIP
 | | | 24 | left ring fingertip | 45 | right ring fingertip
 | | | 25 | left pinky finger MCP | 46 | right pinky finger MCP
 | | | 26 | left pinky finger PIP | 47 | right pinky finger PIP
 | | | 27 | left pinky finger DIP | 48 | right pinky finger DIP
 | | | 28 | left pinky fingertip | 49 | right pinky fingertip
 
## Sing semantics
Based on our prior work (https://link.springer.com/chapter/10.1007/978-3-642-23538-2_42) we can define several 
"independent" semantic features and compute their presence in a video of a sign.

### Location
Represents the location, relative to some fixed body-part, where the main part of the sign occurs. Each SL defines its
own locations (by the means of linguists). Unfortunately, the location of articulation does not need to be a discriminative
feature of the signs. But we shall pretend it is.

The presence of a location feature in a video is based on an analysis of a histogram of hand locations. The location can
be defined per hand.

Universally we can define the following locations:

- **Neutral space** - when there is no predominant location - this is relevant only on the level of a sign not a video 
(eg. each repetition of the same sign happens in different location). Could be a fallback location?
- **Above head**
- **Upper part of the face** - think of top of the head or forehead, even eyebrows.
- **Eyes**
- **Nose**
- **Mouth**
- **Lower part of the face** - as in chin
- **Cheeks**
- **Ears**
- **Neck**
- **Shoulders**
- **Chest**
- **Waist**
- **Arm**
- **Wrist** - we should make this simply "the other hand"

### Movement
Movement is linguistically very complex and combines the movement of all the body parts. In this work, we should focus
only on hand/arm movement. Maybe we can further disentangle hands and arms? Only short-term movements are considered
(eg. "up", "down" movement is ok; "up then down" is not ok)

- **Up-wards**
- **Down-wards**
- **To the right**
- **To the left**
- **Moving away from** - we can define each hand movement in terms of from which (stationary) body part it moves away
- **Moving to**
- **Hold** - basically when the hand stops moving
- **Passage** - e.g. when a hand moves in front of other body parts - should we use this? how?
- **Contact** - important, but how to detect contact? It is hard to distinguish contact and overlap in 2D
- **Twisting** - when wrist is stationary, but the hand moves?

### Orientation
How is the hand oriented.

- **Up-wards**
- **Down-wards**
- **To the body**
- **Away from the body**
- **To the right** - what is right?
- **To the left** - what is left? 

