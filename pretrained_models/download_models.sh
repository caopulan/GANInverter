root="pretrained_models"

function download_model() {
  if [ ! -d "$root/$1" ]; then
    mkdir "$root/$1"
  fi
  if [ -f "$root/$1/$2" ]; then
    echo "文件 $2 已经存在，不需要下载"
  else
    wget "$3" -O "$root/$1/$2"
    echo "已下载文件：$2"
  fi
}

# 如果没有指定任何参数，则下载所有文件
if [ $# -eq 0 ]; then
  echo "没有指定任何参数，将下载所有文件..."
  download_model "generator" "stylegan2-ffhq-config-f.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/stylegan2-ffhq-config-f.pt"
  download_model "e4e" "e4e_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/e4e_ffhq_r50_wp_official.pt"
  download_model "hfgi" "hfgi_ffhq_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hfgi_ffhq_official.pt"
  download_model "hyperstyle" "hyperstyle_ffhq_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hyperstyle_ffhq_official.pt"
  download_model "hyperstyle" "hyperstyle_ffhq_r50_w_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hyperstyle_ffhq_r50_w_official.pt"
  download_model "lsap" "lsap_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/lsap_ffhq_r50_wp_official.pt"
  download_model "other" "model_ir_se50.pth" "https://github.com/caopulan/GANInverter/releases/download/v0.1/model_ir_se50.pth"
  download_model "psp" "psp_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/psp_ffhq_r50_wp_official.pt"
  download_model "psp" "psp_ffhq_r50_w_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/psp_ffhq_r50_w_official.pt"
  download_model "restyle" "restyle-e4e_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/restyle-e4e_ffhq_r50_wp_official.pt"
  echo "所有文件下载完成..."
  exit 0
fi

# 下载指定的文件
for type in "$@"; do
  if [ "$type" == "generator" ]; then
    download_model "generator" "stylegan2-ffhq-config-f.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/stylegan2-ffhq-config-f.pt"
  fi
  if [ "$type" == "e4e" ]; then
    download_model "e4e" "e4e_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/e4e_ffhq_r50_wp_official.pt"
  fi
  if [ "$type" == "hfgi" ]; then
    download_model "hfgi" "hfgi_ffhq_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hfgi_ffhq_official.pt"
  fi
  if [ "$type" == "hyperstyle" ]; then
    download_model "hyperstyle" "hyperstyle_ffhq_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hyperstyle_ffhq_official.pt"
    download_model "hyperstyle" "hyperstyle_ffhq_r50_w_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/hyperstyle_ffhq_r50_w_official.pt"
  fi
  if [ "$type" == "lsap" ]; then
    download_model "lsap" "lsap_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/lsap_ffhq_r50_wp_official.pt"
  fi
  if [ "$type" == "other" ]; then
    download_model "other" "model_ir_se50.pth" "https://github.com/caopulan/GANInverter/releases/download/v0.1/model_ir_se50.pth"
  fi
  if [ "$type" == "psp" ]; then
    download_model "psp" "psp_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/psp_ffhq_r50_wp_official.pt"
    download_model "psp" "psp_ffhq_r50_w_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/psp_ffhq_r50_w_official.pt"
  fi
  if [ "$type" == "restyle" ]; then
    download_model "restyle" "restyle-e4e_ffhq_r50_wp_official.pt" "https://github.com/caopulan/GANInverter/releases/download/v0.1/restyle-e4e_ffhq_r50_wp_official.pt"
  fi
done
