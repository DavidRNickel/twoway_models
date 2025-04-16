pwd

python3 testing.py --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 5 \
    --save-dir "tmp" --log-file 'test_results' --loadfile "tmp/tmp.pt" \
    --num-test-epochs 5000 --num-valid-epochs 1000 --d-model 32 --batch-size 10000