import sys
import time
import threading
import json
from configparser import ConfigParser
from collections import OrderedDict


_section2encoder = {
        'learning': int,
        'information': float,
        'regularizers': str,
        'scores': str}


def cfg2model_settings(cfg_file):
    """
    Returns a mapping ie\n
    {
        'learning': OrderedDict({'nb_topics': 20, 'collection_passes': 50, 'document_passes': 1}),\n
        'information': OrderedDict({'bakground-topics-percentage': 0.1}),\n
        'regularizers': OrderedDict({'smooth-phi': 'smp', 'sparse-theta': 'spt'}),\n
        'scores': OrderedDict({'perplexity': 'per', 'topic-kernel-0.80': 'tk80', 'topic-kernel-0.25': 'tk1'})}\n
    :param str cfg_file: full path
    :return: the parsed settings
    :rtype: OrderedDict
    """
    config = ConfigParser()
    config.read(u'{}'.format(cfg_file))
    # return OrderedDict([(section.encode('utf-8'), OrderedDict([(setting_name.encode('utf-8'), _section2encoder[section](value)) for setting_name, value in config.items(section) if value])) for section in config.sections()])
    return OrderedDict([(section, OrderedDict(
        [(setting_name, _section2encoder[section](value)) for setting_name, value in
         config.items(section) if value])) for section in config.sections()])


class GenericTopicNamesBuilder:
    def __init__(self, nb_topics=0, background_topics_pct=0.0):
        self._nb_topics = nb_topics
        self._background_pct = background_topics_pct
        self._background_topics, self._domain_topics = [], []
        self._names = []

    def _query_topics(self, a_range):
        return [self._names[i] for i in range(*a_range)]

    @property
    def _generic_topic_names(self):
        return ['top_{}'.format(index) for index in [x if len(str(x)) > 1 else '0'+str(x) for x in range(self._nb_topics)]]

    def define_topics(self, names_list):
        self._background_topics, self._domain_topics = [], []
        self._nb_topics = len(names_list)
        self._names = names_list
        return self

    def define_nb_topics(self, nb_topics):
        if nb_topics != self._nb_topics:
            self._background_topics, self._domain_topics = [], []
            self._nb_topics = nb_topics
            self._names = self._generic_topic_names
        return self
    def define_background_pct(self, pct):
        if pct != self._background_pct:
            self._background_topics, self._domain_topics = [], []
            self._background_pct = pct
        return self

    def get_background_topics(self):
        if not self._background_topics:
            self._background_topics = self._query_topics([int(self._background_pct * self._nb_topics)])
        return self._background_topics
    def get_domain_topics(self):
        if not self._domain_topics:
            self._domain_topics = self._query_topics([int(self._background_pct * self._nb_topics), self._nb_topics])
        return self._domain_topics
    def get_background_n_domain_topics(self):
        return self.get_background_topics(), self.get_domain_topics()
    def get_all_topics(self):
        return self.get_background_topics() + self.get_domain_topics()


generic_topic_names_builder = GenericTopicNamesBuilder()


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)
