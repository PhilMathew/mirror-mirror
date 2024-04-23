python generate_forget_set.py -d CIFAR10 -c 9 -n 7000 -o output/test_cifar10  
python train_m1_m3.py -d CIFAR10 -f output/test_cifar10/datasets/forget_set.csv -o output/test_cifar10 -ne 30 -bs 128
python unlearn_forget_set.py -d CIFAR10 -u ssd -f output/test_cifar10/datasets/forget_set.csv -m1 output/test_cifar10/m1/m1_state_dict.pt -o output/test_cifar10 -bs 64
python eval_membership_inference.py -d CIFAR10 -mia logreg -f output/test_mnist/datasets/forget_set.csv -m2 output/test_cifar10/m2/m2_state_dict.pt -m3 output/test_cifar10/m3/m3_state_dict.pt -o output/test_cifar10 -bs 128
