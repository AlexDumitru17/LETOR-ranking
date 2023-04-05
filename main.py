import subprocess
import concurrent.futures
from functools import partial

ranker_configs = {
    1: {'epoch': 50, 'lr': 0.001, 'node': 5},  # RankNet
    4: {'r': 3, 'i': 20},  # Coordinate Ascent
    5: {'epoch': 50, 'lr': 0.001, 'node': 5}, # LambdaRank
    # 6: {'tree': 300, 'leaf': 10},  # LambdaMART
    # Other rankers can use default parameters
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

    # with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
    #     for line in process.stdout:
    #         print(line.strip())

    return_code = process.wait()
    if return_code != 0:
        print(f"RankLib process exited with code {return_code}")

if __name__ == "__main__":
    metrics = ["NDCG@10", "MAP", "P@10"]
    # Number of algorithms
    n = 8
    # TODO: Change fold nr
    fold_nr = 3
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for metric in metrics:
            for model_nr in range(0, n):
                training_data = f"data/mslr_web/Fold{fold_nr}/train.txt"
                vali_data = f"data/mslr_web/Fold{fold_nr}/vali.txt"
                model_output = f"models/Model{model_nr}Metric{metric}Fold{fold_nr}"
                ranker_config = ranker_configs.get(model_nr, {})
                executor.submit(train_ranklib_model, training_data, model_output, model_nr, metric, vali_data, ranker_config)
    # training_data = "data/mslr_web/Fold1/train.txt"
    # test_data = "data/mslr_web/Fold1/test.txt"
    # vali_data = "data/mslr_web/Fold1/vali.txt"
    # model_output = "models/testModel.txt"
    #
    # model_type = 3  # For example, use RankNet
    # metric = "NDCG@10"

    # train_ranklib_model(training_data, model_output, model_type, metric, vali_data)
