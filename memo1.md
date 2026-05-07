Vevo_Timbre
#Params of Flow Matching model: 337.69 M
#Params of Vocoder model: 255.04 M
#Params of Content-Style Tokenizer: 44.29 M

Vevo_Style
#Params of AR model: 481.49 M
#Params of Flow Matching model: 337.69 M
#Params of Vocoder model: 255.04 M
#Params of Content Tokenizer: 58.72 M
#Params of Content-Style Tokenizer: 44.29 M

Vevo_Voice
#Params of AR model: 481.49 M
#Params of Flow Matching model: 337.69 M
#Params of Vocoder model: 255.04 M
#Params of Content Tokenizer: 58.72 M
#Params of Content-Style Tokenizer: 44.29 M

Vevo_TTS
#Params of AR model: 743.30 M
#Params of Flow Matching model: 337.69 M
#Params of Vocoder model: 255.04 M
#Params of Content-Style Tokenizer: 44.29 M

##
TTS는 espeak-ng 문제가 자주 터짐. espeak-ng를 쓰는 PHONEMIZER_ESPEAK_PATH를 공개적으로 지정해주고 실험 돌려야함.
export PHONEMIZER_ESPEAK_LIBRARY=/scratch/$USER/espeak_local/lib/libespeak-ng.so
export PHONEMIZER_ESPEAK_PATH=/scratch/$USER/espeak_local/bin/espeak-ng

(vevo) 342% [x3397a09@glogin03 Amphion]$ REAL_SO_PATH=$(find /scratch/$USER/espeak_local -name "libespeak-ng.so" | head -n 1)
(vevo) 343% [x3397a09@glogin03 Amphion]$ REAL_BIN_PATH=$(find /scratch/$USER/espeak_local -name "espeak-ng" | head -n 1)
(vevo) 344% [x3397a09@glogin03 Amphion]$ echo "찾은 라이브러리 경로: $REAL_SO_PATH"
찾은 라이브러리 경로: /scratch/x3397a09/espeak_local/lib64/libespeak-ng.so
(vevo) 345% [x3397a09@glogin03 Amphion]$ echo "찾은 실행파일 경로: $REAL_BIN_PATH"
찾은 실행파일 경로: /scratch/x3397a09/espeak_local/include/espeak-ng
(vevo) 346% [x3397a09@glogin03 Amphion]$ PHONEMIZER_ESPEAK_LIBRARY=$REAL_SO_PATH PHONEMIZER_ESPEAK_PATH=$REAL_BIN_PATH python -m models.vc.vevo.infer_vevotts
##


학습 순서
1. Tokenizer 학습
# Content Tokenizer (Vocab = 32)
sh egs/codec/vevo/fvq32.sh
    - Adam Optimizer(Initial LR : 1e-4, LR Scheduler : Constant, Epoch = 5000, num_worker = 16, batch size = 32)



# Content-Style Tokenizer (Vocab = 8192)
sh egs/codec/vevo/fvq8192.sh
    - Adam Optimizer(Initial LR : 1e-4, LR Scheduler : Constant, Epoch = 5000, num_worker = 16, batch size = 32)

1.5 Tokenizer 2개 Path 수정
    paths in the egs/vc/AutoregressiveTransformer/ar_conversion.json:

2. Auto-regressive Transformer 학습
Run the following script:
    sh egs/vc/AutoregressiveTransformer/ar_conversion.sh
        - Adam Optimizer(Initial LR : 1e-4, LR Scheduler : Inverse Sqrt, Epoch = 5000, num_worker = 8, batch size = 10)


Similarly, you can run the following script for Vevo-TTS training:
    sh egs/vc/AutoregressiveTransformer/ar_synthesis.sh

2.5 Transformer Path 수정
    egs/vc/FlowMatchingTransformer/fm_contentstyle.json:

3. Flow-matching Transformer
sh egs/vc/FlowMatchingTransformer/fm_contentstyle.sh

