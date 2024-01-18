import logging
import os.path
import shutil
from multiprocessing.pool import Pool
from pathlib import Path, PurePosixPath
from unittest import TestCase

import config
from remote import toko_machine
import utils
from remote.slurm_machine import SLURMMachine

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
        toko: SLURMMachine = config.MACHINES()['mini']
        local_dir = Path(LOCAL_TEST_FILE_DIR.format(copy_style))
        toko_dir = PurePosixPath(TOKO_TEST_FILE_DIR.format(copy_style))
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.mkdir(local_dir)
        local_files: list[Path] = [local_dir / LOCAL_TEST_FILE.format(i) for i in range(10)]
        # Toko file is the same as local
        toko_files: list[PurePosixPath] = [toko_dir / LOCAL_TEST_FILE.format(i) for i in range(10)]
        for local_file in local_files:
            utils.write_local_file(local_file, TEST_FILE_CONTENT)
        copy_result: bytes = toko.cp_to(local_dir, toko_dir, is_folder=True)
        cat_results: list[str] = toko.read_multiple_files(toko_files)
        assert [cat_result == TEST_FILE_CONTENT for cat_result in cat_results], f"cat_results: {cat_results}"
        assert copy_result == b'', f"[{copy_style}] copy_result: {copy_result}"

        # Try to copy again to assess that a new folder is not created inside
        copy_result: bytes = toko.cp_to(local_dir, toko_dir, is_folder=True)
        list_result: list[str] = toko.ls(toko_dir)
        assert len(list_result) == 10, f"[{copy_style}] list_result: {list_result}"

        for local_file in local_files:
            os.remove(local_file)
        os.rmdir(local_dir)
        toko.rm(*toko_files)
        toko.remove_dir(toko_dir)
        try:
            list_result: list[str] = toko.ls(toko_dir)
            assert False, f"[{copy_style}] list_result: {list_result}"
        except FileNotFoundError:
            pass

    @staticmethod
    def _test_copy_style_single(copy_style: str):
        toko: SLURMMachine = config.MACHINES()['mini']
        local_file: Path = Path(LOCAL_TEST_FILE.format(copy_style))
        toko_file: PurePosixPath = PurePosixPath(TOKO_TEST_FILE.format(copy_style))
        if os.path.exists(local_file):
            os.remove(local_file)
        utils.write_local_file(local_file, TEST_FILE_CONTENT)
        copy_result: bytes = toko.cp_to(local_file, toko_file)
        cat_result: str = toko.read_file(toko_file)
        assert cat_result == TEST_FILE_CONTENT, f"cat_result: {cat_result}"
        assert copy_result == b'', f"copy_result: {copy_result}"
        os.remove(local_file)
        toko.rm(toko_file)
        try:
            cat_result: str = toko.read_file(toko_file)
            assert False, f"cat_result: {cat_result}"
        except FileNotFoundError:
            pass
