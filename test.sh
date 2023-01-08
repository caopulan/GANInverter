for i in {-10..10}
do
  CUDA_VISIBLE_DEVICES=4 \
  python scripts/edit.py \
  -c configs/e4e/e4e_ffhq_r50.yaml \
  --inverse_mode encoder \
  --edit_mode ganspace \
  --edit_path editings/ganspace_pca/ffhq_pca.pt \
  --ganspace_directions 23 3 6 "$i"
done