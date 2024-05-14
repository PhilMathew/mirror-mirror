mkdir output/mnist_results
for class in 0 1 2 3 4 5 6 7 8 9 "random"
do
    res_dir=output/mnist_results/forget_${class}
    mkdir $res_dir

    # Generate forget set
    python generate_forget_set.py -d MNIST -c $class -n 7000 -o $res_dir

    # Train original and control models
    python train_original_and_control.py -d MNIST -f ${res_dir}/datasets/forget_set.csv -o $res_dir -ne 3 -bs 128

    # Obtain unlearned model by running unlearning on original
    python unlearn_forget_set.py \
        -d MNIST \
        -u ssd \
        -f ${res_dir}/datasets/forget_set.csv \
        -ckpt ${res_dir}/original/original_state_dict.pt \
        -o $res_dir \
        -bs 64 
    
    # Evaluate membership inference performance
    python eval_membership_inference.py \
        -d MNIST \
        -mia logreg \
        -f ${res_dir}/datasets/forget_set.csv \
        -u ${res_dir}/unlearn/unlearn_state_dict.pt \
        -c ${res_dir}/control/control_state_dict.pt \
        -o $res_dir \
        -bs 64
done

python aggregate_results.py \
    -d output/mnist_results \
    -dt logreg_mia \
    -o output/mnist_results
