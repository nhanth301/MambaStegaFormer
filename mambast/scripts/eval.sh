
python main.py --style_dir /home/nhan/Desktop/Eval/sty/resize --content_dir /home/nhan/Desktop/Eval/cnt/resize     \
                --vgg ../checkpoints/vgg_normalised.pth \
                --decoder_path ../checkpoints/decoder_iter_160000.pth --embedding_path ../checkpoints/embedding_iter_160000.pth \
                --mamba_path ../checkpoints/mamba_iter_65000.pth \
                --output_dir results/ --seed 123456 --mode eval \
                --rnd_style --img_size 512 --d_state 16 --use_pos_embed