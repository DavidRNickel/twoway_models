pwd

echo "snr1=1 snr2=20 k6m2t6"
python3 testing.py --save-dir "tmp" --log-file "test_results.txt" --loadfile "tmp/tmp.pt" \
    --num-valid-epochs 1000 --num-test-epochs 25000 \
    --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 20