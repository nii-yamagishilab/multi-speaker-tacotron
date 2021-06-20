# ==============================================================================
# Copyright (c) 2018-2019, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """


import subprocess
from extensions.phoneset.phoneset import Phoneset


class Flite:

    def __init__(self, binary_path, phoneset_path, args=["-ps"]):
        self.binary_path = binary_path
        self.args = args
        self._phone_set = Phoneset(phoneset_path)

    def command(self, arg):
        return [self.binary_path] + self.args + [arg, 'none']

    def convert_to_phoneme(self, text):
        ## fix for single word utterances
        if len(text.split(' ')) == 1:
            text += ' '
        ## get rid of unicode
        text2 = ''
        for char in text:
            if ord(char) not in range(128):
                continue
            else:
                text2 += char
                
        command = self.command(text2)
        result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        phone_txt = result.stdout.decode('utf-8', 'strict')
        phone_list = phone_txt.split(' ')
        phone_list = phone_list[:-1] if phone_list[-1] == '\n' else phone_list
        phone_ids = []
        for p in phone_list:
            try:
                newid = self._phone_set.phone_to_id(p)
                phone_ids.append(newid)
            except:
                print("BAD PHONE ID: " + p + " IN UTTERANCE: [" + text + "].  SKIPPING PHONE.")
            
        return phone_ids, phone_txt

