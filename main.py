import subprocess
import concurrent.futures
from functools import partial

ranker_configs = {
    1: {'epoch': 50, 'lr': 0.001, 'node': 5},  # RankNet
    4: {'r': 3, 'i': 20},  # Coordinate Ascent
    5: {'epoch': 50, 'lr': 0.001, 'node': 5}, # LambdaRank

}

def train_ranklib_model(training_data, model_output, model_type, metric, validation_data, ranker_config):
    ranklib_jar = "RankLib-2.18.jar"
    ranker_params = " ".join([f"-{key} {value}" for key, value in ranker_config.items()])
    command = f"java -jar {ranklib_jar} -train {training_data} -save {model_output} -ranker {model_type} {ranker_params} -metric2t {metric} -validate {validation_data}"
    print(f"Executing command: {command}")

    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line.strip())

    return_code = process.wait()
    if return_code != 0:
        print(f"RankLib process exited with code {return_code}")


def test_ranklib_model(model_to_load, test_data, metric):
    ranklib_jar = "RankLib-2.18.jar"
    command = f"java -jar {ranklib_jar} -load {model_to_load}  -test {test_data} -metric2T {metric}"
    print(f"Executing command: {command}")

    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line.strip())

    return_code = subprocess.wait()
    if return_code != 0:
        print(f"RankLib process exited with code {return_code}")

def train_test_pltr(loss, dataset):
    command = f"python3 ../PLTR_v2/main.py --loss {loss} --coll_name {dataset} --data_folder ../LETOR_data/{dataset}/"
    with subprocess.Popen(command, shell=True, stdout=open('output_pltr.txt', "w+"), stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line.strip())

if __name__ == "__main__":
    args = []
    datasets = ['MQ2007', 'MQ2008', 'MSLR-WEB10K']
    losses = ['KL_B', 'KL_G', 'KL_B_H', 'KL_G_H']
    for dataset in datasets:
        for loss in losses:
            train_test_pltr(loss, dataset)
    

    metrics = ["NDCG@10", "MAP", "P@10"]
    # Number of algorithms
    n = 8
    # TODO: Change fold nr
    fold_nr = 5
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for metric in metrics:
            for model_nr in range(0, n+1):
                training_data = f"data/istella-s-letor/sample/train.txt"
                vali_data = f"data/istella-s-letor/sample/vali.txt"
                model_output = f"models/Model{model_nr}Metric{metric}Istella"
                ranker_config = ranker_configs.get(model_nr, {})
                executor.submit(train_ranklib_model, training_data, model_output, model_nr, metric, vali_data, ranker_config)
