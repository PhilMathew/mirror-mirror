mkdir output/cifar10_results
for class in 0 1 2 3 4 5 6 7 8 9 "random"
do
    res_dir=output/cifar10_results/forget_${class}
    mkdir $res_dir

    # Generate forget set
    python generate_forget_set.py -d CIFAR10 -c $class -n 7000 -o $res_dir

    # Train original and control models
    python train_original_and_control.py -d CIFAR10 -f ${res_dir}/datasets/forget_set.csv -o $res_dir -ne 200 -bs 128 -lr 0.1

    # Obtain unlearned model by running unlearning on original
    python unlearn_forget_set.py \
        -d CIFAR10 \
        -u ssd \
        -f ${res_dir}/datasets/forget_set.csv \
        -ckpt ${res_dir}/original/original_state_dict.pt \
        -o $res_dir \
        -bs 64 

    # Evaluate membership inference performance
    python eval_membership_inference.py \
        -d CIFAR10 \
        -mia logreg \
        -f ${res_dir}/datasets/forget_set.csv \
        -u ${res_dir}/unlearn/unlearn_state_dict.pt \
        -c ${res_dir}/control/control_state_dict.pt \
        -o $res_dir \
        -bs 64
done

python aggregate_mia_results.py \
    -d output/cifar10_results \
    -mia logreg \
    -o output/cifar10_results