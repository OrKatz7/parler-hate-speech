cd ../src
python3 main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
python3 main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
python3 main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_baseline_classification/ --classification
python3 main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
python3 main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
python3 main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_baseline_classification/ --classification
python3 main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_toxigen_backtranslate_classification/ --back_translation --pretrain_hate --classification
python3 main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_backtranslate_classification/ --back_translation --classification
python3 main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_baseline_classification/ --classification
