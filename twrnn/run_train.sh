pwd

echo "SNR1=1"
python3 main.py --save-dir "tmp" --log-file 'train_log_snr1' --save-freq 10 \
    --tot-N-bits 2 --N-bits 2 --N-channel-use 6 --SNR2 1 --SNR1 1 \
    --num-epochs 100 --use-tensorboard False
