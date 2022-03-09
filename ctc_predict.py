import tensorflow as tf
import ctc_utils
import cv2
import numpy as np
import simpleaudio as sa
import wavio
import tensorflow.compat.v1 as tf_v1
import numpy as np
from midi.player import *
import segment

tf.compat.v1.disable_eager_execution()

i = segment.totalimages
SEMANTIC = ''

model = "Models/semantic_model.meta"
voc_file = "Data/vocabulary_semantic.txt"

tf_v1.reset_default_graph()
sess = tf_v1.InteractiveSession()

# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf_v1.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf_v1.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf_v1.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf_v1.nn.ctc_greedy_decoder(logits, seq_len)

open("output/notes.txt", "w").close()

for x in range (1,i):

    imagep = 'temp/segment'+str(x)+'.jpg'
    print(imagep)

    image = cv2.imread(imagep, 0)
    image = ctc_utils.resize(image, HEIGHT)
    image = ctc_utils.normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],-1,1)

    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

    prediction = sess.run(decoded,
                      feed_dict={
                          input: image,
                          seq_len: seq_lengths,
                          rnn_keep_prob: 1.0,
                      })

    str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    for w in str_predictions[0]:
        print (int2word[w]),
        print ('\t'),
        file1 = open("output/notes.txt","a")
        file1.writelines(int2word[w]+"\t")
        file1.close

    # form string of detected musical notes
    
    for w in str_predictions[0]:
        SEMANTIC += int2word[w] + '\n'


if __name__ == '__main__':

    # gets the audio file
    audio = get_sinewave_audio(SEMANTIC)
    # horizontally stacks the freqs    
    audio =  np.hstack(audio)
    # normalizes the freqs
    audio *= 32767 / np.max(np.abs(audio))
    #converts it to 16 bits
    audio = audio.astype(np.int16)
    # Saves a .wav output file
    wavio.write("output\output.wav", audio, 44100, sampwidth=2)
    #plays midi 
    play_obj = sa.play_buffer(audio, 1, 2, 44100)
    #outputs to the console
    if play_obj.is_playing():
        print("\nplaying...")
        print(f'\n{SEMANTIC}')  
    #stop playback when done
    play_obj.wait_done()
