for class in 0 1 2 3 4 5 6 7 8 9 "random"
do
    res_dir=output/cifar10_results/forget_${class}
    
    # Evaluate membership inference performance
    python eval_perturbation_kld.py \
        -d CIFAR10 \
        -p gaussian \
        -f ${res_dir}/datasets/forget_set.csv \
        -om ${res_dir}/original/original_state_dict.pt \
        -u ${res_dir}/unlearn/unlearn_state_dict.pt \
        -c ${res_dir}/control/control_state_dict.pt \
        -o $res_dir \
        -bs 64
done

python aggregate_results.py \
    -d output/cifar10_results \
    -dt kld_random_gaussian \
    -o output/cifar10_results