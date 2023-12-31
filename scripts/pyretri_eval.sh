echo "=== MAKING DATA JSON ==="
python libs/PyRetri/main/make_data_json.py \
  -d ~/data/COCO/CBIR_data/CBIR_test/class/query/ \
  -sp ~/data/COCO/CBIR_data/CBIR_test/class/query.json \
  -t general
python libs/PyRetri/main/make_data_json.py \
  -d ~/data/COCO/CBIR_data/CBIR_test/class/gallery/ \
  -sp ~/data/COCO/CBIR_data/CBIR_test/class/gallery.json \
  -t general
python libs/PyRetri/main/make_data_json.py \
  -d ~/data/COCO/CBIR_data/CBIR_test/superclass/query/ \
  -sp ~/data/COCO/CBIR_data/CBIR_test/superclass/query.json \
  -t general
python libs/PyRetri/main/make_data_json.py \
  -d ~/data/COCO/CBIR_data/CBIR_test/superclass/gallery/ \
  -sp ~/data/COCO/CBIR_data/CBIR_test/superclass/gallery.json \
  -t general

echo "=== EXTRACTING FEATURES FOR CLASS ==="
python libs/PyRetri/main/extract_feature.py \
  -dj ~/data/COCO/CBIR_data/CBIR_test/class/query.json \
  -sp ~/data/COCO/CBIR_data/CBIR_test/class/features/query \
  -cfg ~/data/COCO/CBIR_data/CBIR_test/class/config.yaml
python libs/PyRetri/main/extract_feature.py \
  -dj ~/data/COCO/CBIR_data/CBIR_test/class/gallery.json \
  -sp ~/data/COCO/CBIR_data/CBIR_test/class/features/gallery \
  -cfg ~/data/COCO/CBIR_data/CBIR_test/class/config.yaml

echo "=== EXTRACTING FEATURES FOR SUPERCLASS ==="
python libs/PyRetri/main/extract_feature.py \
  -dj ~/data/COCO/CBIR_data/CBIR_test/superclass/query.json \
  -sp ~/data/COCO/CBIR_data/CBIR_test/superclass/features/query \
  -cfg ~/data/COCO/CBIR_data/CBIR_test/superclass/config.yaml
python libs/PyRetri/main/extract_feature.py \
  -dj ~/data/COCO/CBIR_data/CBIR_test/superclass/gallery.json \
  -sp ~/data/COCO/CBIR_data/CBIR_test/superclass/features/gallery \
  -cfg ~/data/COCO/CBIR_data/CBIR_test/superclass/config.yaml

echo "=== INDEXING CLASS ==="
python libs/PyRetri/main/index.py -cfg ~/data/COCO/CBIR_data/CBIR_test/class/config.yaml

echo "=== INDEXING SUPERCLASS ==="
python libs/PyRetri/main/index.py -cfg ~/data/COCO/CBIR_data/CBIR_test/superclass/config.yaml