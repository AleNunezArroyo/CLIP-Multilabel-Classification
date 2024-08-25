pip install -U "huggingface_hub[cli]"
pip install transformers
pip install git+https://github.com/openai/CLIP.git

if [ ! -d model/clip-vit-base-patch16 ]; then
    mkdir -p model/clip-vit-base-patch16
fi 

if [ ! -f model/clip-vit-base-patch16/config.json ]; then
    huggingface-cli download \
      openai/clip-vit-base-patch16 \
      --local-dir model/clip-vit-base-patch16 \
      --local-dir-use-symlinks False
fi

