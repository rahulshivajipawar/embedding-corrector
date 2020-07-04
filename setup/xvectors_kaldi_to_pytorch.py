import torch
import subprocess


def get_xvectors_and_ids(kaldi_ark_file):
    """
    Returns a dictionary of xvectors with key being utterance name and value being the tensor.

    Arguments:
        kaldi_ark_file (str): Path to kaldi .ark file

    Returns:
        xvectors_dict (dict): Dictionary of x-vectors with key being utterance name and value being the tensor.
    """
    xvectors_dict = dict () # ids will be keys and xvectors will be values in the form of PyTorch tensor
    copy_xvectors_to_textfile = (subprocess.Popen(['/home/rahul/kaldi/src/bin/copy-vector','ark:{}'.format(kaldi_ark_file),'ark,t:temp.ark']))
    copy_xvectors_to_textfile.wait()
    file_content_array = open('temp.ark','r').read().split('\n')
    num_xvectors = len(file_content_array)-1
    for i in range(num_xvectors):
        string_list = file_content_array[i].split(' ')
        xvector_array = string_list[3:len(string_list)-1]
        xvector_tensor = [float(i) for i in xvector_array]
        xvectors_dict[string_list[0]] = xvector_tensor
    return xvectors_dict

def save_tensors(xvectors_dict, target_directory):
    """
    Saves elements of a dictionary with tensor name being the key and the value being the tensor.

    Arguments:

        xvectors_dict (dict): Dictionary of tensor values with keys with utterance name and value being corresponding x-vector value.
        target_directory (str): Path to target directory where tensors are to be saved.
    """

    for utt_name, tensor in xvectors_dict.items():
        torch.save(tensor, target_dictionary + '/' + utt_name + '.pt')

# long_utterances_directory = "/data-ssd/rahul-artifacts/embedding-corrector/exp/xvector_nnet_1a/xvectors_long_utterances"
# target_dictionary = "/home/rahul/projects/embedding-corrector/data/long_utterances_tensors"

#for i in range(1,21):
#    kaldi_file = long_utterances_directory + "/xvector." + str(i) + ".ark"
#    tensor_dictionary = get_xvectors_and_ids(kaldi_file)
#    save_tensors(tensor_dictionary, target_dictionary)

short_utterances_directory = "/data-ssd/rahul-artifacts/embedding-corrector/exp/xvector_nnet_1a/xvectors_short_utterances"
target_dictionary = "/home/rahul/projects/embedding-corrector/data/short_utterances_tensors"

for i in range(1,21):
    kaldi_file = short_utterances_directory + "/xvector." + str(i) + ".ark"
    tensor_dictionary = get_xvectors_and_ids(kaldi_file)
    save_tensors(tensor_dictionary, target_dictionary)
