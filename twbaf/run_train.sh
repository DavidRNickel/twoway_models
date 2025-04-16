pwd

echo "fb_20_m2"
python3 main.py --save-dir "tmp" --log-file "tmp_snr20.txt" --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 20 

echo "fb_30_m2"
python3 main.py --save-dir "tmp" --log-file "tmp_snr30.txt" --K 6 --M 2 --T 6 --snr-ff 1 --snr-fb 30