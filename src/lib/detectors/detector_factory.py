from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  # 'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
