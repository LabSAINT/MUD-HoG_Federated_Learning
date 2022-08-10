#!/bin/bash
declare -a AR_METHOD=("mudhog" "fedavg" "median" "gm" "krum" "mkrum" "foolsgold")

#### LIST CLIENTS for MNIST dataset
# Unrelibale: upto 4 nodes
LIST_UNRELIABLE=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37")

# Untargeted: Sign-flipping attackers up to 5 nodes
LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13")

# Untargeted: Additive noise attackers, upto 6 node
LIST_AN_ATTACKERS=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28,30")

# Targeted: Label-flipping attackers: increasing from 3 nodes to 8 nodes.
LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

# Targeted: Multi-labeli-flipping attackers: increasing from 3 nodes to 8 nodes.
LIST_MLF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

DATASET="fashion_mnist"
LR=1e-2
WD=1e-4
MOMENTUM=0.9
GLOBAL_EPOCHS=40
LOCAL_EPOCHS=4
STD_UR=30

# Series Exp1A (AN_LF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (LF) attackers; where j in range [2,6].
#EXP_NAME="sExp1Bb_40ep" 
#for ((i=0; i < 1; i++));
#do
#for ((j=4; j<6; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --max_std_unreliable $STD_UR &
#done
#done

# sExp2A (AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) multi-label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp2Ba_40ep"
#for ((i=1; i < 2; i++));
#do
#for ((j=2; j<4; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients 40  --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --source_labels 1,2,3
#done
#done


# Series Exp3A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp3Bb_40ep"
#for ((i=0; i < 1; i++));
#do
#for ((j=2; j<3; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS  --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR 
#done
#done

# Series Exp4A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+2) multi-label-flipping (MLF) attackers; where j in range [1,5].
EXP_NAME="sExp4Ba_40ep"
for ((i=0; i < 1; i++));
do
for ((j=2; j<3; j++));
do
CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR
done
done
