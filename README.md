# AI-GPT-2D-Platformer
2D Platformer that has AI generated Levels using Genetic Algorithm and GPT model


# to prepare the data set
run : python data/levels/prepare.py

# to train the model
run : python train.py config/train_levels_char.py
*) you can add parameters in the terminal aswell like :
python train.py config/train_levels_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# to get a sample from the model
run : python sample.py --out_dir=out-shakespeare-char