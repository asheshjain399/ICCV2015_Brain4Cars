# ICCV2015_Brain4Cars
Code for ICCV2015 paper "Car That Knows Before You Do: Anticipating Maneuvers via Learning Temporal Driving Models"


Data set is now available here (16 GB) : https://www.dropbox.com/sh/yndzlk3o90ooq2j/AACWUT8xjabmILM6-rm1_gNAa?dl=0 

###External Dependecies
You will need the ```minFunc``` code from Mark Schmidt. It is available [here](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html). Add it in the root directory of this repo before proceeding. 

###Known Issues
When training AIOHMM, you will sometimes see the message ```Step Direction is illegal!```, and the training might get stuck. Restart the training in this scenario. If the problem persists, revert to an older version of the repo (```ec9681e```).