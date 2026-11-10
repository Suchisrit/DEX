export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
python train.py  \
--train_batch_size 4 \
--datasets \
    irs \
    waymo \
--image_txts \
    splits/irs_images.txt \
    splits/waymo_images.txt \
--intrinsics_paths \
    data/irs_derived/K.npy \
    splits/waymo_intrinsics.txt \
--crop_shapes \
    378 672 \
    420 630 \
--k1_ranges \
    -0.1 -0.02 \
    -0.2 -0.02 \
--k2_ranges \
    -0.1 -0.004 \
    -0.2 -0.004 \
--k3_ranges \
    -0.1 -0.0008 \
    -0.2 -0.0008 \
--k4_ranges \
    -0.05 -0.00003 \
    -0.10 -0.00003 \
--dataset_weights \
    1.0 \
    0.5 \
--n_batches_per_update 2 \
--learning_rates 1e-3 1e-4 \
--learning_schedule 5 10 \
--aug_prob_brightness 0.5 \
--aug_prob_contrast 0.5 \
--aug_prob_saturation 0.5 \
--aug_prob_hue 0.2 \
--aug_prob_blur 0.2 \
--aug_brightness_range 0.6 1.4 \
--aug_contrast_range 0.6 1.4 \
--aug_saturation_range 0.6 1.4 \
--aug_hue_range -0.1 0.1 \
--aug_blur_range 0.1 0.5 \
--num_extenders 12 \
--output_reference_frame distance \
--checkpoint_path checkpoints/Test1 \
--n_step_per_checkpoint 500 \
--n_step_per_summary 100 \
--device cuda \
--n_thread 4 \
--model_name depthanything \