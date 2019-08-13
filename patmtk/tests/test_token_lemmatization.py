import pytest
from processors.string_processors import StringLemmatizer

@pytest.fixture(scope='module')
def lemmatize():
    return StringLemmatizer().process


@pytest.fixture(scope='module')
def str_n_bytes_pair():
    b = b'similar calls have been made across the political spectrum in the aftermath of the shooting, showing subtle signs that the gop-controlled congress is compelled to take some form of action on crafting gun violence legislation.'
    return str(b), b


@pytest.mark.skip(reason="This fails because there is no functionality implemented to thoroughly check and process input before lemmatization.")
class TestLemmatizationBehaviour(object):

    def test_str_n_bytes_result(self, lemmatize, str_n_bytes_pair):
        assert lemmatize(str_n_bytes_pair[0]) == lemmatize(str_n_bytes_pair[1])
