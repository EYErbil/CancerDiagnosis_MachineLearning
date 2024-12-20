# Before starting, 
download the dataset-->https://www.kaggle.com/datasets/egeyigiterbil/cervicalcancer/data
Extract the files to a Directory named CervicalCancer
the file structure should look like:

-->CervicalCancer  
           -------> Herlev  
           -------> Mendeley  
           -------> sipakmed  
--> train.py  
--> dataloader.py  
--> ....  

## To show how we handle data visually:
python dataloader.py --dataset Herlev --classification binary --visualize
## To show with augmentations:
python dataloader.py --dataset Herlev --classification binary --augmentation 2 --visualize

## We can change classification type too:
python dataloader.py --dataset Herlev --classification multiclass --augmentation 0 --visualize

## To train:
python train.py --model_name model1-2-3-4-5-6 --dataset Herlev-Mendeley-sipakmed --classification binary or multiclass --augmentation 0-1-2 --epochs 100
The train script will put the model into a new folder called checkpoints
For no augmentations, I use 0 , and for augmentations, use 2

## To test with all metrics:
python test.py --model_name model1  --checkpoint ./checkpoints/model1_best.pth --dataset Herlev --classification binary --num_visualize 15 --save_dir ./evaluation_results_binary

model_name gets the model architecture, --checkpoint gets the file name, dataset and classification are same as above, num visualize will show visualized results, and the results will be
saved in new folder

## multiclass:
python test.py --model_name model1  --checkpoint ./checkpoints/model1_best.pth --dataset Herlev --classification multiclass--num_visualize 15 --save_dir ./evaluation_results_multiclass

## To Ensemble:
python test_ensemble.py --models model1 model2 model3 --checkpoints ./checkpoints/model1_best.pth ./checkpoints/model2_best.pth ./checkpoints/model3_best.pth --ensemble_method avg_prob --dataset Herlev --classification binary --num_visualize 15 --save_dir ./evaluation_results_ensemble --num_visualize 10


--models gets the architectures, --checkpoints are same as above, but they have to be as same order as --models, 
we have 3 ensembling techniques, avg_prob, max_prob, majority_vote
rest are same
