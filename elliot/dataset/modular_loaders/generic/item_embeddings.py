from types import SimpleNamespace
import pandas as pd
import typing as t

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ItemEmbeddings(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "attribute_file", None)
        self.users = users
        self.items = items
        # DENTRO MAP AVRO IL DIZIONARIO CON ASSOCIATO GLI EMBEDDINGS
        self.map_ = self.load_attribute_file(self.attribute_file)
        self.items = self.items & set(self.map_.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self):
        """QUI NON CI SERVONO PIU LE ULTIME RIGHE"""
        ns = SimpleNamespace()
        ns.__name__ = "ItemEmbeddings"
        ns.object = self
        ns.feature_map = self.map_
        ns.features = None
        ns.nfeatures = None
        ns.private_features = None
        ns.public_features = None
        #ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        #ns.nfeatures = len(ns.features)
        #ns.private_features = {p: f for p, f in enumerate(ns.features)}
        #ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def load_attribute_file(self, attribute_file, separator='\t'):
        map_ = {}
        with open(attribute_file) as file:
            for line in file:
                if "article" in line:
                    continue
                else:
                    line = line.split(separator)
                    item_id = line[0]
                    embedding = [float(el) for el in line[1][1:-2].split(" ")]
                    map_[int(item_id)] = embedding
        return map_
