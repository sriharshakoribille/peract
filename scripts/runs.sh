# Jars, only clip, 10 demos
CUDA_VISIBLE_DEVICES=0 python train.py \
    method=PERACT_BC \
    rlbench.tasks=\[close_jar\] \
    rlbench.task_name='clip_close_jar_p3' \
    rlbench.cameras=\[front,left_shoulder,right_shoulder,wrist\] \
    rlbench.demos=10 \
    rlbench.demo_path=$PERACT_ROOT/data/peract_baseline/train \
    replay.batch_size=4 \
    replay.path=$PERACT_ROOT/replay \
    replay.max_parallel_processes=2 \
    method.voxel_sizes=\[100\] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=1024 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=\[0.0,0.0,45.0\] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=7100\
    framework.num_weights_to_keep=15 \
    framework.start_seed=0 \
    framework.log_freq=500 \
    framework.save_freq=500 \
    framework.logdir=$PERACT_ROOT/logs/10_demos/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    rlbench.episode_length=10 \
    method.dense_clip_sims=True \
    method.no_rgb=True

# Jars, only clip, 100 demos
CUDA_VISIBLE_DEVICES=0 python train.py \
    method=PERACT_BC \
    rlbench.tasks=\[close_jar\] \
    rlbench.task_name='clip_close_jar_512_rls2' \
    rlbench.cameras=\[front,left_shoulder,right_shoulder,wrist\] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/peract_baseline/train \
    replay.batch_size=7 \
    replay.path=$PERACT_ROOT/replay \
    replay.max_parallel_processes=20 \
    method.voxel_sizes=\[100\] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=512 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=\[0.0,0.0,45.0\] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=20100\
    framework.num_weights_to_keep=15 \
    framework.start_seed=0 \
    framework.log_freq=500 \
    framework.save_freq=1000 \
    framework.logdir=$PERACT_ROOT/logs/100_demos/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    rlbench.episode_length=10 \
    method.dense_clip_sims=True \
    method.no_rgb=True

clip_light_bulb_p3
# Screw bulb, only clip, 100 demos
CUDA_VISIBLE_DEVICES=0 python train.py \
    method=PERACT_BC \
    rlbench.tasks=\[light_bulb_in\] \
    rlbench.task_name='clip_light_bulb_512_p3' \
    rlbench.cameras=\[front,left_shoulder,right_shoulder,wrist\] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/peract_baseline/train \
    replay.batch_size=3 \
    replay.path=$PERACT_ROOT/replay \
    replay.max_parallel_processes=25 \
    method.voxel_sizes=\[100\] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=512 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=\[0.0,0.0,45.0\] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=40100\
    framework.num_weights_to_keep=25 \
    framework.start_seed=0 \
    framework.log_freq=500 \
    framework.save_freq=1000 \
    framework.logdir=$PERACT_ROOT/logs/100_demos/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    rlbench.episode_length=12 \
    method.dense_clip_sims=True \
    method.no_rgb=True \
    framework.load_existing_weights=True

# Multi task, jar + bulb, only clip, 100 demos
CUDA_VISIBLE_DEVICES=0 python train.py \
    method=PERACT_BC \
    rlbench.tasks=\[close_jar,light_bulb_in\] \
    rlbench.task_name='clip_multi_jar_bulb_rls2' \
    rlbench.cameras=\[front,left_shoulder,right_shoulder,wrist\] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/peract_baseline/train \
    replay.batch_size=4 \
    replay.path=$PERACT_ROOT/replay \
    replay.max_parallel_processes=30 \
    method.voxel_sizes=\[100\] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=\[0.0,0.0,45.0\] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=30100\
    framework.num_weights_to_keep=51 \
    framework.start_seed=0 \
    framework.log_freq=500 \
    framework.save_freq=500 \
    framework.logdir=$PERACT_ROOT/logs/100_demos/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    rlbench.episode_length=12 \
    method.dense_clip_sims=True \
    method.no_rgb=True \
    framework.load_existing_weights=True

# Jar, clip and rgb, 100 demos
CUDA_VISIBLE_DEVICES=0 python train.py \
    method=PERACT_BC \
    rlbench.tasks=\[close_jar\] \
    rlbench.task_name='clip_rgb_close_jar_p3' \
    rlbench.cameras=\[front,left_shoulder,right_shoulder,wrist\] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/peract_baseline/train \
    replay.batch_size=2 \
    replay.path=/tmp/replay \
    replay.max_parallel_processes=2 \
    method.voxel_sizes=\[100\] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=\[0.0,0.0,45.0\] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=7100\
    framework.num_weights_to_keep=15 \
    framework.start_seed=0 \
    framework.log_freq=500 \
    framework.save_freq=500 \
    framework.logdir=$PERACT_ROOT/logs/100_demos/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    rlbench.episode_length=10 \
    method.dense_clip_sims=True \
    method.no_rgb=False