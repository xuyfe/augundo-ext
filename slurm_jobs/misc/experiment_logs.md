### training and eval unos - current pipeline (w/ horizontal flop, temporal flop, resize, and color/brightness)
--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.0634,     0.9367,      4.405,      0.141,      7.421,      0.952,      0.980,      0.989

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
    1.3512,     0.0698,     0.2700,     0.0742 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
    1.2432,     0.0596,     0.4320,     0.0686 

### training and eval unos - old pipeline, WITHOUT color jitterness and WITHOUT resize, ONLY horizontal flips (100k iterations)

The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_unos_new/final/unos_model.pth
CWD:            /home/ox4
Restored UnOS model from: /home/ox4/augundo-ext/checkpoints/augundo_unos_new/final/unos_model.pth (step 100000)

--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.0728,     0.9745,      4.511,      0.158,      8.516,      0.941,      0.972,      0.985

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
    1.5982,     0.0698,     0.9257,     0.0852 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
    1.6678,     0.0597,     0.9570,     0.0805 

Evaluation completed

### training and eval unos - old pipeline WITH color jitterness AND resize (300k iterations)
The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth
CWD:            /home/ox4
Restored UnOS model from: /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth (step 300000)

--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.1126,     1.2930,      5.683,      0.206,     23.704,      0.866,      0.945,      0.975

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
    2.8249,     0.2314,     0.5054,     0.2370 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
    4.1936,     0.3235,     0.7021,     0.3326 

Evaluation completed


### training and eval unos - old
The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth
CWD:            /home/ox4
Restored UnOS model from: /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth (step 100000)

--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.2018,     3.3971,      7.778,      0.297,     39.744,      0.769,      0.893,      0.947

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
    4.6974,     0.3873,     0.9690,     0.3974 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
    6.4271,     0.4245,     0.9904,     0.4376 

Evaluation completed

### training and eval unos - old pipeline

The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth
CWD:            /home/ox4
Restored UnOS model from: /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth (step 100000)

--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    0.2240,     5.0988,      8.303,      0.315,     37.101,      0.787,      0.903,      0.952

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
    4.5097,     0.3599,     0.9931,     0.3710 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
    7.0867,     0.4789,     0.9946,     0.4908 

Evaluation completed

### training and eval bdf - old pipeline
The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_bdf/final/bdf_model.pth
Results dir:    /home/ox4/augundo-ext/results/augundo_bdf
CWD:            /home/ox4
Restored BDF model from: /home/ox4/augundo-ext/checkpoints/augundo_bdf/final/bdf_model.pth (step 217260)
Loaded 200 test stereo pairs
Inference complete. 200 samples
Disparities saved to: /home/ox4/augundo-ext/results/augundo_bdf/disparities.npy

Evaluating against ground truth...

   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    2.3600,   122.7384,     37.609,      6.606,     95.822,      0.044,      0.088,      0.130
Evaluation completed

### training and eval unos - old pipeline
The following have been reloaded with a version change:
  1) CUDA/12.9.1 => CUDA/12.6.0

Checkpoint:     /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth
CWD:            /home/ox4
Restored UnOS model from: /home/ox4/augundo-ext/checkpoints/augundo_unos/final/unos_model.pth (step 100000)

--- Evaluating on kitti_2015 (200 images) ---

Depth metrics (KITTI 2015):
   abs_rel,     sq_rel,        rms,    log_rms,     d1_all,         a1,         a2,         a3
    6.0735,   421.9951,     64.982,      1.894,    100.000,      0.009,      0.027,      0.056

Disparity metrics (KITTI 2015):
       epe,   noc_rate,   occ_rate,   err_rate 
   34.4126,     1.0000,     0.9950,     1.0000 


--- Evaluating on kitti_2012 (194 images) ---

Disparity metrics (KITTI 2012):
       epe,   noc_rate,   occ_rate,   err_rate 
   40.1319,     1.0000,     0.9948,     1.0000 

Evaluation completed