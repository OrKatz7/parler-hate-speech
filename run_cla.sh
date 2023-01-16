cd src

/home/kaor/.conda/envs/mmdet/bin/python main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_baseline_classification/ --classification


/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_baseline_classification/ --classification


/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
/home/kaor/.conda/envs/mmdet/bin/python main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_baseline_classification/ --classification