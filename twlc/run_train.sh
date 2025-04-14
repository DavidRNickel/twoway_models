pwd

echo "SNR1=1"
python3 main.py --save-dir "twoway/k2m2t6/snr1_1/snr2_1" --log-file 'train_log' --save-freq 10 \
    --K 2 --M 2 --T 6 --snr-ff 1 --snr-fb 1 \
    --batch-size 100000 --num-epochs 100 --grad-clip .5  --use-tensorboard True --d-model 32

python3 main.py --save-dir "twoway/k2m2t6/snr1_1/snr2_5" --log-file 'train_log' --save-freq 10 \
    --K 2 --M 2 --T 6 --snr-ff 1 --snr-fb 5 \
    --batch-size 100000 --num-epochs 100 --grad-clip .5  --use-tensorboard True --d-model 32

python3 main.py --save-dir "twoway/k2m2t6/snr1_1/snr2_10" --log-file 'train_log' --save-freq 10 \
    --K 2 --M 2 --T 6 --snr-ff 1 --snr-fb 10 \
    --batch-size 100000 --num-epochs 100 --grad-clip .5  --use-tensorboard True --d-model 32

python3 main.py --save-dir "twoway/k2m2t6/snr1_1/snr2_15" --log-file 'train_log' --save-freq 10 \
    --K 2 --M 2 --T 6 --snr-ff 1 --snr-fb 15 \
    --batch-size 100000 --num-epochs 100 --grad-clip .5  --use-tensorboard True --d-model 32
