import numpy as np
from history_attention_utils import grid_rows, locate_visual_segments, semantic_token_indices, roi_mask_xyxy

class T:
    def __init__(self, x): self.x=x
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.x

def test_grid_and_exact_segments():
    grids=grid_rows(T([[1,4,6],[1,2,2]]), 2)
    got=locate_visual_segments(T([[9,99,99,99,99,99,99,8,99]]),99,grids,[{"kind":"history"},{"kind":"current"}])
    assert [(x['sequence_start'],x['sequence_end'],x['grid_height'],x['grid_width']) for x in got] == [(1,7,2,3),(8,9,1,1)]

def test_semantic_and_roi():
    assert semantic_token_indices(['{',' CLICK',':',' 123','}']) == [1,3]
    assert roi_mask_xyxy([0,0,50,50],2,2,(100,100)).sum()==1
