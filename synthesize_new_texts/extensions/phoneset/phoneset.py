# ==============================================================================
# Copyright (c) 2018-2019, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import json


class Phoneset:

    def __init__(self, phoneset_path):
        self._build_phone_map(phoneset_path)

    def _build_phone_map(self, phoneset_path):
        with open(phoneset_path, mode='r') as f:
            parsed = json.load(f)
            self._phone_to_id = {item["phone"]: item["id"] for item in parsed["phones"]}
            self._id_to_phone = {item["id"]: item["phone"] for item in parsed["phones"]}

    def phone_to_id(self, phone):
        return self._phone_to_id[phone]
        
    def id_to_phone(self, an_id):
        return self._id_to_phone[an_id]
