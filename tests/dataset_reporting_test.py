import os
import pytest


class TestDatasetReporter(object):

    def test_simple_strings(self, dataset_reporter):
        strings_list1 = dataset_reporter.get_infos(details=False)
        assert '\n'.join(strings_list1) == 'unittest-dataset/200_lowercase_monospace_unicode_deaccent_normalize-lemmatize_minlength-2_maxlength-25_ngrams-1_nobelow-1_noabove-0.5_weight-counts_format1-uci.format2-vowpal.pkl'
        assert len(strings_list1) == len(os.listdir(dataset_reporter._r))

    def test_detailed_strings(self, dataset_reporter):
        strings_list2 = dataset_reporter.get_infos(details=True)
        assert len(strings_list2) == len(os.listdir(dataset_reporter._r))

        strings_list2 = dataset_reporter.get_infos(details=True, selection='unittest-dataset')
        assert len(strings_list2) == 1
