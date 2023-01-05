cd src
python main.py --model Narrativaai/deberta-v3-small-finetuned-hate_speech18 --outputs_dir ../outputs_baseline/
python main.py --model microsoft/deberta-v3-base --outputs_dir ../outputs_baseline/
python main.py --model microsoft/deberta-v3-large --outputs_dir ../outputs_baseline/
python main.py --model microsoft/deberta-v2-xlarge --outputs_dir ../outputs_baseline/
python main.py --model microsoft/deberta-v2-large --outputs_dir ../outputs_baseline/
