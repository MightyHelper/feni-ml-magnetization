import logging
import os.path
import shutil
from multiprocessing.pool import Pool
from unittest import TestCase

import config
import toko_utils
import utils

LOCAL_TEST_FILE = "test_{}.txt"
TOKO_TEST_FILE = "~/test_{}.txt"
LOCAL_TEST_FILE_DIR = "tests_data_{}"
TOKO_TEST_FILE_DIR = "~/tests_data_{}"
TEST_FILE_CONTENT = "test"


class TestTokoUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=config.LOG_LEVEL)

    def test_copy_single_file_to_toko(self):
        with Pool(processes=2) as pool:
            pool.map(TestTokoUtils._test_copy_style_single, ['rsync', 'scp'])

    def test_copy_multi_file_to_toko(self):
        with Pool(processes=2) as pool:
            pool.map(TestTokoUtils._test_copy_style_multi, ['rsync', 'scp'])

    @staticmethod
    def _test_copy_style_multi(copy_style: str):
        config.TOKO_COPY_SCRIPT = copy_style
        local_dir = LOCAL_TEST_FILE_DIR.format(copy_style)
        toko_dir = TOKO_TEST_FILE_DIR.format(copy_style)
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.mkdir(local_dir)
        local_files = [os.path.join(local_dir, LOCAL_TEST_FILE.format(i)) for i in range(10)]
        # Toko file is the same as local
        toko_files = [os.path.join(toko_dir, LOCAL_TEST_FILE.format(i)) for i in range(10)]
        for local_file in local_files:
            utils.write_local_file(local_file, TEST_FILE_CONTENT)
        copy_result: bytes = toko_utils.TokoUtils.copy_file_to_toko(local_dir, toko_dir, is_folder=True)
        cat_results: list[str] = toko_utils.TokoUtils.read_multiple_files(toko_files)
        assert [cat_result == TEST_FILE_CONTENT for cat_result in cat_results], f"cat_results: {cat_results}"
        assert copy_result == b'', f"[{copy_style}] copy_result: {copy_result}"

        # Try to copy again to assess that a new folder is not created inside
        copy_result: bytes = toko_utils.TokoUtils.copy_file_to_toko(local_dir, toko_dir, is_folder=True)
        list_result: list[str] = toko_utils.TokoUtils.list_files(toko_dir)
        assert len(list_result) == 10, f"[{copy_style}] list_result: {list_result}"

        for local_file in local_files:
            os.remove(local_file)
        os.rmdir(local_dir)
        toko_utils.TokoUtils.remove_files(*toko_files)
        toko_utils.TokoUtils.remove_dir(toko_dir)
        try:
            list_result: list[str] = toko_utils.TokoUtils.list_files(toko_dir)
            assert False, f"[{copy_style}] list_result: {list_result}"
        except FileNotFoundError:
            pass

    @staticmethod
    def _test_copy_style_single(copy_style: str):
        config.TOKO_COPY_SCRIPT = copy_style
        local_file = LOCAL_TEST_FILE.format(copy_style)
        toko_file = TOKO_TEST_FILE.format(copy_style)
        if os.path.exists(local_file):
            os.remove(local_file)
        utils.write_local_file(local_file, TEST_FILE_CONTENT)
        copy_result: bytes = toko_utils.TokoUtils.copy_file_to_toko(local_file, toko_file)
        cat_result: str = toko_utils.TokoUtils.read_file(toko_file)
        assert cat_result == TEST_FILE_CONTENT, f"cat_result: {cat_result}"
        assert copy_result == b'', f"copy_result: {copy_result}"
        os.remove(local_file)
        toko_utils.TokoUtils.remove_files(toko_file)
        try:
            cat_result: str = toko_utils.TokoUtils.read_file(toko_file)
            assert False, f"cat_result: {cat_result}"
        except FileNotFoundError:
            pass
