pwd

echo "m2 ff_1"
echo "snr1=1 snr2=1 k6m2t6"
# python3 testing.py --save-dir "letter/snr1_1/m2/snr2_1" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_1_m2/20240712-044233.pt" \
#     --num-valid-epochs 1000 --num-test-epochs 10000 \
#     --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 1 

# echo "snr1=1 snr2=5 k6m2t6"
# python3 testing.py --save-dir "letter/snr1_1/m2/snr2_5" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_5_m2/20240712-141947.pt" \
#     --num-valid-epochs 1000 --num-test-epochs 10000 \
#     --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 5 

echo "snr1=1 snr2=10 k6m2t6"
python3 testing.py --save-dir "letter/snr1_1/m2/snr2_10" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_10_m2/20240713-084110.pt" \
    --num-valid-epochs 1000 --num-test-epochs 10000 \
    --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 10

echo "snr1=1 snr2=15 k6m2t6"
python3 testing.py --save-dir "letter/snr1_1/m2/snr2_15" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_15_m2/20240715-153855.pt" \
    --num-valid-epochs 1000 --num-test-epochs 10000 \
    --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 15 

echo "snr1=1 snr2=20 k6m2t6"
python3 testing.py --save-dir "letter/snr1_1/m2/snr2_20" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_20_m2/20240712-142533.pt" \
    --num-valid-epochs 1000 --num-test-epochs 25000 \
    --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 20

echo "snr1=1 snr2=30 k6m2t6"
python3 testing.py --save-dir "letter/snr1_1/m2/snr2_30" --log-file "test_results.txt" --loadfile "ff_1/m2/fb_30_m2/20240720-210929.pt" \
    --num-valid-epochs 1000 --num-test-epochs 25000 \
    --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 30