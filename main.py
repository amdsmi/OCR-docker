import os
import cv2
import shutil
import argparse
import pandas as pd
import seaborn as sns
from subprocess import Popen
from evaluate import evaluate
import matplotlib.pyplot as plt
if os.path.exists('dataset/config.py'):
    print('your config is detect')
    os.remove('config/config.py')
    shutil.copy('dataset/config.py', 'config/config.py')
else:
    print('you are using custom config')
import config.config as cfg


class OcrDocker:

    def __init__(self, report_template, dataset_path, report_path):

        self.report_template = report_template
        self.report_path = report_path
        self.dataset = dataset_path
        self.dataset_checker()

    def path_checker(self, subset):

        return True if os.path.exists(self.dataset + subset + '/images') and os.path.exists(
            self.dataset + subset + '/labels.csv') else False

    def dataset_checker(self):

        if self.path_checker('train'):
            print("train data is ok")
        else:
            print('train data is not right')

        if self.path_checker('validation'):
            print("validation data is ok")
        else:
            print('validation data is not detect ')

        if self.path_checker('test'):
            print("your train data is ok")
        else:
            print('your train data is not right')

    def metrics_plot(self, best, train_table, loss_path='output/loss_plot.png', acc_path='output/acc_plot.png'):

        best_result = train_table[train_table['epoch'] == int(best[3:]) - 1]

        if self.path_checker('validation'):

            train_table.set_index('epoch')[['loss', 'val_loss']].plot.line().get_figure().savefig(
                loss_path)
            train_table.set_index('epoch')[
                ['ctc_loss_accuracy', 'val_ctc_loss_accuracy']].rename(columns={"ctc_loss_accuracy": "accuracy", "val_ctc_loss_accuracy": "val_accuracy"}).plot.line().get_figure().savefig(acc_path)

            return best_result['loss'].values[0], best_result['ctc_loss_accuracy'].values[0], best_result['val_loss']. \
                values[0], best_result['val_ctc_loss_accuracy'].values[0]

        else:

            train_table.set_index('epoch')[['loss']].plot.line().get_figure().savefig(loss_path)
            train_table.set_index('epoch')[
                ['ctc_loss_accuracy']].plot.line().get_figure().savefig(acc_path)

            return best_result['loss'].values[0], best_result['ctc_loss_accuracy'].values[0], 0.0, 0.0

    @staticmethod
    def trainer(models_path='runs/exp/saved_models/', train_result_path='runs/exp/csv_log/result.csv'):

        train = Popen(['python3', 'train.py'])
        _, _ = train.communicate()

        epochs = os.listdir(models_path)
        best_model_path = 'sm-0'
        for epoch in epochs:
            if int(epoch[3:]) > int(best_model_path[3:]):
                best_model_path = epoch

        train_result = pd.read_csv(train_result_path)

        return best_model_path, train_result

    @staticmethod
    def convert_onnx(best_model_path, save_path='output/ocr_best.onnx'):
        process = Popen(
            ['python3', 'convert_to_onnx.py', '--model',best_model_path, '--output_name', save_path])
        _, _ = process.communicate()

    @staticmethod
    def tester(model_onnx='output/ocr_best.onnx',
               test_path='dataset/test/images',
               label_path='dataset/test/labels.csv',
               char_path='runs/exp/char_list/characters.txt',
               use_gpu=False):

        test_result = evaluate(model=model_onnx,
                               dataset=test_path,
                               label=label_path,
                               characters=char_path,
                               use_gpu=use_gpu)

        strict = round(test_result['strict'], 3)
        similarity = round(test_result['similarity'], 3)
        print(f'the strict accuracy is: {strict}')
        print(f'the similarity accuracy is: {similarity}')
        return strict, similarity

    @staticmethod
    def statistics_calculator(imgs_path, result_path):

        _, _, files = next(os.walk(imgs_path))

        size = []
        width = []
        height = []
        ratio = []
        for file in files:
            size.append(os.stat(imgs_path + '/' + file).st_size)
            img = cv2.imread(imgs_path + '/' + file)
            width.append(img.shape[0])
            height.append(img.shape[1])
            ratio.append(img.shape[0] / img.shape[1])
        df = pd.DataFrame({'SIZE': size, 'WIDTH': width, 'HEIGHT': height, 'RATIO': ratio})
        df = df.describe().round(decimals=2).T[['mean', 'std', 'min', 'max']]
        fig = plt.figure(facecolor='w', edgecolor='k')
        sns.heatmap(df.head(), annot=True, cmap='viridis', cbar=False)
        plt.savefig(result_path)

    def metrics_report_writer(self):

        base = open(self.report_template, 'r')
        old_report = base.readlines()
        base.close()
        new_report = old_report[:9]

        new_report.append(old_report[9].replace('Input_w', str(cfg.input_w)))
        new_report.append(old_report[10].replace('Input_h', str(cfg.input_h)))
        new_report.append(old_report[11].replace('Batch_size', str(cfg.batch_size)))
        new_report.append(old_report[12].replace('DownSample_factor', str(cfg.downsample_factor)))
        new_report.append(old_report[13].replace('Epochs', str(cfg.epochs)))
        new_report.append(old_report[14].replace('Save_freq', str(cfg.save_freq)))
        new_report.append(old_report[15].replace('Patience', str(cfg.patience)))
        new_report.append(old_report[16].replace('Lr', str(cfg.lr)))
        new_report.append(old_report[17].replace('Decay', str(cfg.decay)))
        new_report.append(old_report[18].replace('Momentum', str(cfg.momentum)))

        for row in old_report[19:26]:
            new_report.append(row)
        if self.path_checker('train'):
            train_size = os.listdir(self.dataset + 'train/images/')
            new_report.append(old_report[26].replace('train_sample_number', f'{len(train_size)}'))
            self.statistics_calculator(self.dataset + 'train/images/', 'output/train.png')
            new_report.append('![train](train.png)\n')

        if self.path_checker('validation'):
            validation_size = os.listdir(self.dataset + 'validation/images/')
            new_report.append(old_report[27].replace('val_sample_number', f'{len(validation_size)}'))
            self.statistics_calculator(self.dataset + 'validation/images/', 'output/validation.png')
            new_report.append('![test](validation.png)\n')

        if self.path_checker('test'):
            test_size = os.listdir(self.dataset + 'test/images/')
            new_report.append(old_report[28].replace('test_sample_number', f'{len(test_size)}'))
            self.statistics_calculator(self.dataset + 'test/images/', 'output/test.png')
            new_report.append('![test](test.png)\n')

        for row in old_report[29:37]:
            new_report.append(row)

        model_path, train_table = self.trainer()

        train_loss, train_accuracy, validation_loss, validation_accuracy = self.metrics_plot(model_path, train_table)

        new_report.append(
            old_report[37].replace('loss_plot_path', 'loss_plot.png').replace('acc_plot_path', 'acc_plot.png'))
        for i in old_report[38:41]:
            new_report.append(i)

        new_report.append(
            old_report[41].replace('train_loss', str(round(train_loss, 4))).replace('val_loss',
                                                                                    str(round(validation_loss, 4))))
        new_report.append(
            old_report[42].replace('train_acc', str(round(train_accuracy, 4))).replace('val_acc',
                                                                                       str(round(validation_accuracy,
                                                                                                 4))))

        self.convert_onnx('runs/exp/saved_models/' + model_path)

        shutil.copy('predict.py', 'output/predict.py')
        shutil.copy('runs/exp/char_list/characters.txt', 'output/characters.txt')

        if self.path_checker('test'):
            strict, similarity = self.tester()

            for row in old_report[43:48]:
                new_report.append(row)

            new_report.append(old_report[48].replace('test_res', str(strict)))
            new_report.append(old_report[49].replace('test_sim', str(similarity)))

        file = open(self.report_path, "w")
        file.writelines(new_report)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_template', type=str, default='row_readme.md', help='')
    parser.add_argument('--dataset_path', type=str, default='dataset/', help='')
    parser.add_argument('--report_path', type=str, default='output/report.md', help='')
    args = parser.parse_args()
    ocr_docker = OcrDocker(report_template=args.report_template, dataset_path=args.dataset_path,
                           report_path=args.report_path)
    ocr_docker.metrics_report_writer()
