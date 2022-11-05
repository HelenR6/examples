allAttacks=("l2_0.15" "l2_3" "linf1_1020" "linf4_255")
for model_name in "$@"
do
    for attack in ${allAttacks[@]};
        do
            python examples/imagenet/adv_main.py -a $model_name --attack $attack -b 128 --evaluate --pretrained /content/gdrive/MyDrive/imagenet
        done
    python imagenet-r/eval.py $model_name
    python natural-adv-examples/eval.py $model_name
    python examples/imagenet/sketch_main.py -a $model_name -b 64 --evaluate --pretrained /content/gdrive/MyDrive/sketch

done
