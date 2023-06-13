mkdir data
cd data
wget "https://chmura.put.poznan.pl/s/PwhipofrbYnr5xu/download" -O cones.zip
unzip cones
mv data dataset_color
cd ..
mkdir dataset_YOLO
cd dataset_YOLO
wget "https://chmura.put.poznan.pl/s/ZnUceQgv3IVYw3Y/download" -O YOLO_Dataset.zip
unzip YOLO_Dataset.zip
wget "https://chmura.put.poznan.pl/s/Qz2dnYB5yIQpJ1Z/download" -O train.csv
wget "https://chmura.put.poznan.pl/s/OqUfxGWRxHD5UBd/download" -O val.csv
