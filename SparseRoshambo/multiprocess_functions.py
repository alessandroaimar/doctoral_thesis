import cv2
import numpy as np

def get_image(path, output_size=None, normalize=None, shift=None, move_channel_first=False, images_datatype=np.float32):

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if np.amax(image) == 0:
        print("ERROR - Read image max value is 0 - {}".format(path))

    if output_size is not None:
        image = cv2.resize(image, output_size)

    if normalize is not None:
        image = image / normalize

    if shift is not None:
        image = image - shift

    if move_channel_first is True:
        image = np.swapaxes(image, 0, 2)

    image = np.array(image,images_datatype)
    return image


def gen_augmented_data(base_images, base_labels, queue):
    from augmentation import augment_images
    import numpy as np
    from math import ceil
    import gc
    from multiprocessing import Process, Queue


    num_images = len(base_images)
    num_parallel_processes = 32

    num_images_groups = ceil(num_images / num_parallel_processes)

    # generate a shuffle ordering augment_image(image,label)

    process_queues = list()
    processes = list()

    #print("Initialization of pool of augmenter...")

    for group_idx in range(num_parallel_processes):
        new_queue = Queue(2)
        process_queues.append(new_queue)

        new_process = Process(target=augment_images, args=(
            base_images[group_idx * num_images_groups: min(num_images, (group_idx + 1) * num_images_groups)],
            base_labels[group_idx * num_images_groups: min(num_images, (group_idx + 1) * num_images_groups)],
            np.random.randint(0, 2 ** 31 - 1),
            process_queues[group_idx]))

        new_process.start()
        processes.append(new_process)

    #print("Initialization done")

    while True:
        all_augmented_images = list()
        all_augmented_labels = list()

        for worker_queue in process_queues:
            augmented_data = worker_queue.get()

            all_augmented_images.append(augmented_data["images"])
            all_augmented_labels.append(augmented_data["labels"])

        # for each image we create a modified copy as well as keep the original one
        #all_augmented_images_return = np.concatenate([base_images] + all_augmented_images)
        #all_augmented_labels_return = np.concatenate([base_labels] + all_augmented_labels)

        queue_dict = {"labels": all_augmented_labels, "images": all_augmented_images}
        queue.put(queue_dict)

        #print("proceeding multiprocess")

