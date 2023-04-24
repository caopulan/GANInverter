if [ "$1" = "e4e" ]; then
  url="https://example.com/models/model_a_weights.pth"
elif [ "$1" = "psp" ]; then
  url="https://example.com/models/model_b_weights.pth"
elif [ "$1" = "restyle" ]; then
  url="https://example.com/models/model_c_weights.pth"
elif [ "$1" = "hyperstyle" ]; then
  url="https://example.com/models/model_d_weights.pth"
elif [ "$1" = "hfgi" ]; then
  url="https://example.com/models/model_e_weights.pth"
elif [ "$1" = "lsap" ]; then
  url="https://example.com/models/model_f_weights.pth"
else
  echo "无效的参数：$1"
  exit 1
fi

wget "$url" -O "$1/$2.pt"
echo "已下载模型权重：$url"
