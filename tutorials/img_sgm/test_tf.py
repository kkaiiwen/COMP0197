import os
import time

import tensorflow as tf
from PIL import Image

from loader import H5ImageLoader
import utils_tf as utils
from network_tf import ResUNet


DATA_PATH = './data'


## trained model
save_path = "results_tf"
epoch = 99
seg_net_imported = ResUNet(init_ch=16)
weights_path = os.path.join(save_path, f"epoch{epoch}.weights.h5")


## test data
loader_test = H5ImageLoader(DATA_PATH+'/images_test.h5', 20, DATA_PATH+'/labels_test.h5')


## compute test results
losses_all, dsc_scores_all, inf_times = [], [], []
first_batch = True

for frames_test, masks_test in loader_test:

    t0 = time.time()
    frames_test, masks_test = utils.pre_process(frames_test, masks_test)

    if first_batch:
        _ = seg_net_imported(frames_test, training=False)
        seg_net_imported.load_weights(weights_path)
        first_batch = False

    predicts_test = seg_net_imported(frames_test, training=False)
    inf_times += [time.time()-t0]

    losses_all += [utils.dice_loss(predicts_test, masks_test)]
    dsc_scores_all += [utils.dice_binary(predicts_test, masks_test)]

print('val-loss={:0.5f}, val-dice={:0.5f}, inference-time={:.3f}sec'.format(
    tf.reduce_mean(tf.concat(losses_all,axis=0)),
    tf.reduce_mean(tf.concat(dsc_scores_all,axis=0)),
    sum(inf_times)/len(inf_times)
    ))


# visualise the last batch
img_size = list(frames_test.shape)
image_montage = Image.fromarray(tf.reshape(tf.cast(frames_test,tf.uint8),[-1]+img_size[2:]).numpy())
image_montage.save("test_images.jpg")
label_montage = Image.fromarray(tf.reshape(masks_test>0.5,[-1]+img_size[2:3]).numpy())
label_montage.save("test_labels.jpg")
predict_montage = Image.fromarray(tf.reshape(predicts_test>0.5,[-1]+img_size[2:3]).numpy())
predict_montage.save("test_predicted.jpg")
