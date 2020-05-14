FILES=/content/corpus/*

for f in $FILES
do
  python ./bert/create_pretraining_data.py \
    --input_file=$f \
    --output_file=./created/pretraining_corpus_${f:9:8}.tfrecord \
    --vocab_file=./base/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5
done
