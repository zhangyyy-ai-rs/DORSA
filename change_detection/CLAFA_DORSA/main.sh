CUDA_VISIBLE_DEVICES=0 python main.py --backbone dorsa_t --dataset LEVIR_256_split --checkpoint_dir ./checkpoints/dorsa_t/LEVIR_e200
CUDA_VISIBLE_DEVICES=0 python main.py --backbone dorsa_t --dataset WHU_256 --checkpoint_dir ./checkpoints/dorsa_t/WHU_256_e200
CUDA_VISIBLE_DEVICES=0 python main.py --backbone dorsa_t --dataset CDD_256 --checkpoint_dir ./checkpoints/dorsa_t/CDD_256_e200
CUDA_VISIBLE_DEVICES=0 python main.py --backbone dorsa_t --dataset SYSU_256 --checkpoint_dir ./checkpoints/dorsa_t/SYSU_256_e200
