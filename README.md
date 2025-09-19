# SegVol: Universal and Interactive Volumetric Medical Image Segmentation Adaptation Implementation
Implementation for the adapted SegVol method for use in the IS_Validation_Framework by Esmaeili et al. Most of the implementation follows the logic utilised in their SegFM methodology. 

We also have an implementation for supporting point-based evaluations despite the fact that they did not make a submission for point based methods, since this was provided in the original implementation! We also use the original checkpoint to avoid any complications associated with image normalisation that would arise from the pre-normalisation employed by the SegFM challenge organisers. 


While the bounding box implementation will follow their method (i.e., using image arrays and resampling etc.) for the point based methods we will avoid using such lossy methods. In this circumstance we perform an exact 1-to-1 mapping of the points into the varying zoom-in, zoom-out domains where possible. The only information loss occurs with the sliding window mechanism which does not support background points.

Lastly, for multiple interaction iterations (for point based editing), we use atomic inference as no previous segmentations are used as inputs. In this circumstance, the entire history of points is used simultaneously for inference. 

