mkdir output/mnist_results
for class in 0 1 2 3 4 5 6 7 8 9
do
    res_dir=output/mnist_results/forget_${class}
    mkdir $res_dir

    # Generate forget set
    python generate_forget_set.py -d MNIST -c $class -n 7000 -o $res_dir

    # Train M1 and M3
    python train_m1_m3.py -d MNIST -f ${res_dir}/datasets/forget_set.csv -o $res_dir -ne 3 -bs 128

    # Obtain M2 by running unlearning on M1
    python unlearn_forget_set.py \
        -d MNIST \
        -u ssd \
        -f ${res_dir}/datasets/forget_set.csv \
        -m1 ${res_dir}/m1/m1_state_dict.pt \
        -o $res_dir \
        -bs 64 
    
    # Evaluate membership inference performance
    python eval_membership_inference.py \
        -d MNIST \
        -mia logreg \
        -f ${res_dir}/datasets/forget_set.csv \
        -m2 ${res_dir}/m2/m2_state_dict.pt \
        -m3 ${res_dir}/m3/m3_state_dict.pt \
        -o $res_dir \
        -bs 64
done
