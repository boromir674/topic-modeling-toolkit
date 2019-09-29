import os
import pytest

# @pytest.fixture(scope='module')
# def senario():

import subprocess

# from click.testing import CliRunner

this_dir = os.path.dirname(os.path.realpath(__file__))

def test_launching(tmpdir):
    target_directory = str(tmpdir.mkdir('unittest-datasets-dir'))
    os.environ["COLLECTIONS_DIR"] = target_directory
    ro = subprocess.run(['transform', 'posts', os.path.join(this_dir, 'test-pipeline.cfg'), '--sample 100', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['train', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['tune', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['make-graphs', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['report-datasets', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['report-models', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['report-topics', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0
    ro = subprocess.run(['report-kl', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ro.returncode == 0

#     def test_main():
#         runner = CliRunner()
#         result = runner.invoke(main, ['--help'])
#         assert result.exit_code == 0
