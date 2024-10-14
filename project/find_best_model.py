# log_file_path = "models/data_aug_80by20/1/slurm_NS.783.out"
# log_file_path = "models/data_aug_80by20/2/slurm_NS.788.out"
# log_file_path = "models/data_aug_80by20/3/slurm_NS.789.out"
# log_file_path = "models/data_aug_80by20/4/slurm_NS.816.out"
# log_file_path = "models/data_aug_80by20/5/slurm_NS.880.out"
log_file_path = "models/data_aug_80by20/6/slurm_NS.822.out"

best_loss = float('inf')
idx_best_loss = 0

def extract_loss(log_line):
    if log_line.find("Validation Loss: ") == -1:
        return float('inf')

    start_idx = log_line.find("Validation Loss: ") + len("Validation Loss: ")
    end_idx = log_line.find(",", start_idx)
    return float(log_line[start_idx:end_idx])


with open(log_file_path, 'r') as file:
    for i, line in enumerate(file):
        if extract_loss(line.strip()) < best_loss:
            best_loss = extract_loss(line.strip())
            idx_best_loss = i

# Print the line with best loss and path where it was saved
with open(log_file_path, 'r') as file:
    lines = file.readlines()
    print(lines[idx_best_loss].strip())
    print(lines[idx_best_loss+1].strip())


# O output:

# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.057, Precision: 0.725, Recall: 0.515, F1 Score: 0.602, IoU: 0.431, Acc: 0.980
# Saved model to: models/data_aug_80by20/1/lraspp_193_val_0.06_f1_0.60_f1_train_0.76.pt
# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.056, Precision: 0.835, Recall: 0.589, F1 Score: 0.691, IoU: 0.527, Acc: 0.985
# Saved model to: models/data_aug_80by20/2/fcn_resnet50_341_val_0.06_f1_0.69_f1_train_0.83.pt
# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.060, Precision: 0.746, Recall: 0.625, F1 Score: 0.680, IoU: 0.515, Acc: 0.983
# Saved model to: models/data_aug_80by20/3/fcn_resnet101_343_val_0.06_f1_0.68_f1_train_0.76.pt
# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.071, Precision: 0.684, Recall: 0.521, F1 Score: 0.592, IoU: 0.420, Acc: 0.979
# Saved model to: models/data_aug_80by20/4/deeplabv3_resnet50_80_val_0.07_f1_0.59_f1_train_0.46.pt
# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.074, Precision: 0.716, Recall: 0.354, F1 Score: 0.474, IoU: 0.311, Acc: 0.977
# Saved model to: models/data_aug_80by20/5/deeplabv3_resnet101_122_val_0.07_f1_0.47_f1_train_0.61.pt
# mansur@g14:~/Desktop/NewSpace/project$ python3 find_best_model.py 
# Validation Loss: 0.068, Precision: 0.726, Recall: 0.449, F1 Score: 0.555, IoU: 0.384, Acc: 0.979
# Saved model to: models/data_aug_80by20/6/deeplabv3_mobilenet_v3_large_34_val_0.07_f1_0.56_f1_train_0.51.pt
