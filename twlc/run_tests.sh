pwd

# python3 testing.py --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 1 \
#     --save-dir "twoway/k6m2t6/snr1_n1/snr2_1" --log-file 'test_results' --loadfile "twoway/k2m2t6/snr1_n1/snr2_1/20250312-114852.pt" \
#     --num-test-epochs 5000 --num-valid-epochs 1000 --d-model 32 --batch-size 10000

python3 testing.py --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 5 \
    --save-dir "twoway/k6m2t6/snr1_n1/snr2_5" --log-file 'test_results' --loadfile "twoway/k2m2t6/snr1_n1/snr2_5/20250312-155054.pt" \
    --num-test-epochs 5000 --num-valid-epochs 1000 --d-model 32 --batch-size 10000

# python3 testing.py --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 10 \
#     --save-dir "twoway/k6m2t6/snr1_n1/snr2_10" --log-file 'test_results' --loadfile "twoway/k2m2t6/snr1_n1/snr2_10/20250312-175447.pt" \
#     --num-test-epochs 5000 --num-valid-epochs 1000 --d-model 32 --batch-size 10000

# python3 testing.py --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 15 \
#     --save-dir "twoway/k6m2t6/snr1_n1/snr2_15" --log-file 'test_results' --loadfile "twoway/k2m2t6/snr1_n1/snr2_15/20250312-195915.pt" \
#     --num-test-epochs 5000 --num-valid-epochs 1000 --d-model 32 --batch-size 10000
