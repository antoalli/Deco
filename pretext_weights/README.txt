Pretrained weights for local and global encoder branch. Loaded as encoder weights initialization before completion training.
- Contrastive learning at Global Encoder: 'global_contrastive'
    --ckt from: simCLR_cropRot_ScaleP05_Jitt_sgdCos_DP_4POS_1602780721
    --Num Positive samples: 4
    --Transforms applied: Pointcloud2Partial, PointcloudRotate, PointcloudScale (with prob=0.5), PointcloudJitter
- Classification at Global Encoder: 'global_cla'
- Denoising at Local Encoder (with GPD as local encoder branch as in all paper experiments) : 'gpd_local_denoising'
- Denoising at Local Encoder (DGCNN instead of GPD) : 'dgcnn_local_denoising'