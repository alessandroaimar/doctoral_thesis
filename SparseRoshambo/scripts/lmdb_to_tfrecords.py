import tensorflow as tf
import datetime
import sys
import os
import numpy as np

sys.path.append(r"D:\DL\datasets\Roshambo\lmdb_train\\")
import datum_pb2
import lmdb


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_tfrecords(open_lmdb, save_dir, tf_records_name):
    rows = 64
    cols = 64
    depth = 1

    with open_lmdb.begin() as txn:
        cursor = txn.cursor()
        i = [0, 0, 0, 0]
        now = datetime.datetime.now()
        date = "{}_{}_{}__{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        save_name = tf_records_name + "_" + date
        print("File savename: {}".format(save_name))
        filenames = [os.path.join(save_dir, "class_0_{}.tfrecords".format(save_name)),
                     os.path.join(save_dir, "class_1_{}.tfrecords".format(save_name)),
                     os.path.join(save_dir, "class_2_{}.tfrecords".format(save_name)),
                     os.path.join(save_dir, "class_3_{}.tfrecords".format(save_name))]

        writers = [tf.python_io.TFRecordWriter(filenames[0]),
                   tf.python_io.TFRecordWriter(filenames[1]),
                   tf.python_io.TFRecordWriter(filenames[2]),
                   tf.python_io.TFRecordWriter(filenames[3])]

        print("Starting DB read")
        total_read = 0
        for key, value in cursor:
            datum = datum_pb2.Datum()
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            x = flat_x.reshape(datum.height, datum.width, datum.channels)
            y = datum.label
            image_raw = x.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(y)),
                'image_raw': _bytes_feature(image_raw)}))
            writers[y].write(example.SerializeToString())
            i[y] += 1
            total_read += 1
            if total_read % 10000 == 0:
                print("Image: ", total_read)


        for writer in writers:
            writer.close()

        print("Statistic for classes: {}".format(i))


def open_lmdb_file(file_path):
    lmdb_env = lmdb.open(file_path)
    return lmdb_env


if __name__ == '__main__':
    print("Starting...")
    mode = "train"

    lmdb_path = r"D:\DL\datasets\Roshambo\lmdb_train\\" + mode
    save_dir = r"D:\\DL\\datasets\\Roshambo\\tf_train\\" + mode
    tf_records_name = mode

    lmdb_env = open_lmdb_file(lmdb_path)

    write_to_tfrecords(lmdb_env, save_dir, tf_records_name)
