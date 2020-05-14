!python ./drive/My\ Drive/defense/transform_tf_to_pytorch.py \
  --tf_checkpoint_path=./drive/My\ Drive/defense/rdf_only_bert/rdf_only_bert_model.ckpt \
  --bert_config_file=./drive/My\ Drive/defense/base/bert_config.json \
  --pytorch_dump_path=./drive/My\ Drive/defense/rdf_only_bert/rdf_only_bert_model_pytorch.bin 

!python ./drive/My\ Drive/defense/transform_tf_to_pytorch.py \
  --tf_checkpoint_path=./drive/My\ Drive/defense/rdf_bert/rdf_bert_model.ckpt \
  --bert_config_file=./drive/My\ Drive/defense/base/bert_config.json \
  --pytorch_dump_path=./drive/My\ Drive/defense/rdf_bert/rdf_bert_model_pytorch.bin