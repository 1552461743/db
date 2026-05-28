cd /home/nb666/HybrIK/data/yifu_banshen
source /opt/ros/humble/setup.bash
python3 scripts/bag_to_csv.py /home/nb666/HybrIK/data/yifu_banshen/data/1 --max-delta 50



conda activate hybrik
cd /home/nb666/HybrIK/data/yifu_banshen
python scripts/csv_add_hybrik.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset.csv \
  --output-csv /home/nb666/HybrIK/data/yifu_banshen/data/1/csv_export/synced_dataset2.csv \
  --batch-size 4 \
  --num-workers 2
  
  
cd /home/nb666/HybrIK/data/yifu_banshen
python3 scripts/resize_csv_images.py \
  /home/nb666/HybrIK/data/yifu_banshen/data/1 \
  --num-workers 8
