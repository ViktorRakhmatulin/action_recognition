pushd data/HDM05-15/

# Drops z coordinate from initial dataset.
python ../../parse_data_to_pixel.py \
    HDM05-15-objects-annotations-coords_normPOS.data\
    -s splits/annotated_subsequence_splits/HDM05-15-part1_processed.txt --sf list \
    HDM05-15-part1_dropped.pkl

python ../../parse_data_to_pixel.py \
    HDM05-15-objects-annotations-coords_normPOS.data\
    -s splits/annotated_subsequence_splits/HDM05-15-part2_processed.txt --sf list \
    HDM05-15-part2_dropped.pkl

popd
