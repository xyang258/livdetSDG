
ep=bestA
for run in 1 2 3 4 5
do
    cp checkpoints/maps_seg_run${run}/${ep}_net_Seg.pth checkpoints/eval/
    
    python eval_syn_real.py --datadir /path/to/datasets \
      --datasetname liver \
      --eval_result_folder /path/to/evaluation/results \
      --train_val_test test \
      --filtering True \
      --fix_test True \
      --test_all True \
      --name eval \
      --model eval \
      --input_nc 1 \
      --output_nc 1  \
      --gpu_ids 0 \
      --no_flip \
      --dataset_mode test \
      --model_suffix "Seg" \
      --epoch ${ep} \
      --run_number ${run}
done
