for run in 1 2 3 4 5
do

    python train.py \
        --name maps_seg \
        --model seg \
        --no_html \
        --input_nc 1 \
        --output_nc 1 \
        --gpu_ids 0 \
        --batch_size 1 \
        --norm instance \
        --load_size 286 \
        --crop_size 256 \
        --n_epochs 10 \
        --n_epochs_decay 0 \
        --wt_c 1 \
        --wt_d 1 \
        --run_number ${run}

    cp checkpoints/maps_seg/bestA_net_Seg.pth checkpoints/maps_seg/bestAs1_net_Seg.pth
    cp checkpoints/maps_seg/bestA_EMA_net_Seg.pth checkpoints/maps_seg/bestAs1_EMA_net_Seg.pth
    
    python train.py \
        --name maps_seg \
        --model segs2 \
        --no_html \
        --input_nc 1 \
        --output_nc 1 \
        --gpu_ids 0 \
        --batch_size 1 \
        --norm instance \
        --load_size 286 \
        --crop_size 256 \
        --continue_train \
        --epoch bestAs1 \
        --epoch_count 11 \
        --n_epochs 20 \
        --n_epochs_decay 0 \
        --wt_c 1 \
        --wt_d 1 \
        --run_number ${run}
    mv checkpoints/maps_seg checkpoints/maps_seg_run${run}
done
