import cli_parts.shapefolder as shapefolder
import cli_parts.executions as executions
import unittest


class TestShapefolder(unittest.TestCase):
	def test_ls(self):
		shapefolder.ls()

	def test_shrink(self):
		shapefolder.shrink()


class TestExec(unittest.TestCase):
	def test_ls(self):
		executions.ls()


if __name__ == '__main__':
	unittest.main()
