import os
import tensorflow as tf
import time
from keras.callbacks import TensorBoard
from telepyth import TelepythClient
from datetime import datetime
tp = TelepythClient('14227435377386201718')


class TensorBoardBatchLogger(TensorBoard):
    def __init__(self, project_path, step_size_train, batch_size, log_every=1, VERBOSE=0, **kwargs):
        tf.summary.FileWriterCache.clear()
        self.project_path = project_path
        self.batch_size = batch_size
        self.log_dir = self._create_run_folder()

        super().__init__(log_dir=self.log_dir, batch_size=self.batch_size, **kwargs)
        self.log_every = log_every
        self.counter = 0
        self.sum_loss = 0
        self.epoch_num = 0
        self.counter_for_mean = 1
        self.epoch_end = False
        self.VERBOSE = VERBOSE
        if self.VERBOSE:
            text = '*Start training:' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '*' + '\n'
            tp.send_text(text)

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.epoch_end:
            self.epoch_end = False
            self.counter_for_mean = 1
            self.sum_loss = 0
        self.sum_loss += logs['loss']
        mean_loss = self.sum_loss / self.counter_for_mean
        self.counter_for_mean += 1

        if self.counter % self.log_every == 0:
            logs['mean_loss'] = mean_loss
            logs['train_on_batch_loss'] = logs.pop('loss')
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        if self.VERBOSE:
            text = 'Epoch num: ' + str(self.epoch_num + 1) + '\n' + 'Time for val: ' + str(round(time.time()-self.start, 1)) + '\n'
            text += 'val\_loss: ' + str(round(logs['val_loss'], 2)) + '- val\_acc: ' + str(round(logs['val_acc'], 4)) + '\n'
            tp.send_text(text)

        self.epoch_num += 1
        self.epoch_end = True
        for name, value in logs.items():
            if (name in ['batch', 'size']) or ('val' not in name):
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def _create_run_folder(self):
        #P ATH_TO_LOGS_ON_SERVER = 'C:\\Users\ddenisov\PycharmProjects\cardsmobile_recognition\logs'
        # PATH_TO_LOGS_ON_SERVER = os.path.join(os.path.dirname(os.path.dirname(self.project_path)), 'cardsmobile_data', 'logs')
        PATH_TO_LOGS = os.path.join(self.project_path, 'logs')
        if not os.path.exists(PATH_TO_LOGS):
            os.mkdir(PATH_TO_LOGS)

        temp_path_run = os.path.join(PATH_TO_LOGS, 'run')
        temp_path = temp_path_run + '_1'
        i = 2
        while os.path.exists(temp_path):
            temp_path = temp_path_run + '_' + str(i)
            i += 1
        return temp_path
