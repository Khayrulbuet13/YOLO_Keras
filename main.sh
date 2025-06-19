python main_keras.py --train --input-size 128x256 >debug_log_keras.txt
python main.py --train --input-size 256x128 >debug_log_pytorch.txt 




########
actual


python main.py --train --input-size 128x256 >debug_log_pytorch.txt 
python main_keras.py --train --input-size 128x256 >debug_log_keras.txt