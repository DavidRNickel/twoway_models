pwd

python3 testing.py --tot-N-bits 6 --N-bits 2 --N-channel-use 6 --SNR1 1 --SNR2 5 \
    --save-dir "tmp" --test-log-file 'test_results' --loadfile "tmp/tmp.pt" \
    --num-test-epochs 5000 --num-valid-epochs 1000 --batch-size 10000