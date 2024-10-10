# Inference



You can evaluate the detection model following:

```bash
tools/dist_test.sh projects/configs/PersPETR/persdetr3d_vov_800_bs2_seq_24e.py training_best_model_prompt/new_matching/latest.pth 8 --eval bbox
```

