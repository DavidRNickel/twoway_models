pwd

echo "ff_1"

echo "fb_1"
python3 testing.py --save-dir "ff_1/fb_1_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 1 --loadfile "ff_1/fb_1_m2/20240712-100920.pt" --num-valid-epochs 10000 --num-test-epochs 10000

echo "fb_5"
python3 testing.py --save-dir "ff_1/fb_5_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 5 --loadfile "ff_1/fb_5_m2/20240712-194325.pt" --num-valid-epochs 10000 --num-test-epochs 10000

echo "fb_10"
python3 testing.py --save-dir "ff_1/fb_10_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 10 --loadfile "ff_1/fb_10_m2/20240713-130036.pt" --num-valid-epochs 10000 --num-test-epochs 10000

echo "fb_15"
python3 testing.py --save-dir "ff_1/fb_15_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 15 --loadfile "ff_1/fb_15_m2/20240715-210843.pt" --num-valid-epochs 10000 --num-test-epochs 10000

echo "fb_20"
python3 testing.py --save-dir "ff_1/fb_20_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 20 --loadfile "ff_1/fb_20_m2/20240712-194905.pt" --num-valid-epochs 10000 --num-test-epochs 10000

echo "fb_30"
python3 testing.py --save-dir "ff_1/fb_30_m2" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 30 --loadfile "ff_1/fb_30_m2/20240715-015859.pt" --num-valid-epochs 10000 --num-test-epochs 10000