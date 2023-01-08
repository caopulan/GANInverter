for i in {0..20}
do
  CUDA_VISIBLE_DEVICES=5 \
  python scripts/edit.py \
  -c configs/e4e/e4e_ffhq_r50.yaml \
  --inverse_mode encoder \
  --edit_mode ganspace \
  --edit_path editings/ganspace_pca/ffhq_pca.pt \
  --ganspace_directions "$i" 0 18 2
done