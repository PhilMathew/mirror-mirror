python generate_forget_set.py -d MNIST -c 9 -n 7000 -o output/test_mnist
python train_m1_m3.py -d MNIST -f output/test_mnist/datasets/forget_set.csv -o output/test_mnist -ne 4 -bs 256
python unlearn_forget_set.py -d MNIST -u ssd -f output/test_mnist/datasets/forget_set.csv -m1 output/test_mnist/m1/m1_state_dict.pt -o output/test_mnist -bs 128
python eval_membership_inference.py -d MNIST -mia lira -f output/test_mnist/datasets/forget_set.csv -m2 output/test_mnist/m2/m2_state_dict.pt -m3 output/test_mnist/m3/m3_state_dict.pt -o output/test_mnist/lira_results
