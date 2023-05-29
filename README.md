# Efficient Full Projector Compensation using Natural Images


!(https://github.com/kylin-leo/FullProjectorCompensation/blob/main/fullcompensation.png)

## Dependencies
* pytorch
* numpy
* opencv-python
* raft


## Running code

```
python parameterestimation.py
python compensate.py
```


## Visualization of the parameter estimation

![estgif](https://github.com/kylin-leo/FullProjectorCompensation/blob/main/gif/opt.gif)

Note:

Here the multiplication by 5 is to visualize the minor reconstruction error of our method.


Sign + and - denotes the positive and negative component of \alpha respectively.


## Visualization of the full compensation

![fullcompgif](https://github.com/kylin-leo/FullProjectorCompensation/blob/main/gif/comp.gif)
