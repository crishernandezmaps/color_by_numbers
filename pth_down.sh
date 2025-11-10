rm -f sam_vit_b_01ec64.pth
curl -L --fail --progress-bar \
  -o sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

ls -lh sam_vit_b_01ec64.pth   # ≈ 375 MB