pwd

echo "SNR1=1"
python3 main.py --save-dir "tmp" --log-file 'train_log_snr1' --save-freq 10 \
    --K 6 --M 3 --T 9 --snr-ff 1 --snr-fb 1 \
    --batch-size 50000 --num-epochs 100 --grad-clip .5  --d-model 32

# python3 main.py --save-dir "tmp" --log-file 'train_log_snr5' --save-freq 10 \
#     --K 2 --M 2 --T 6 --snr-ff 1 --snr-fb 5 \
#     --batch-size 100000 --num-epochs 100 --grad-clip .5  --use-tensorboard True --d-model 32