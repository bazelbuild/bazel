import os
import unittest
from src.test.py.bazel import test_base
from tools.ctexplain.bazel_api import BazelApi

# Tests for bazel_api.py.
class BazelApiTest(test_base.TestBase):

    _bazel_api: BazelApi = None
    
    def setUp(self):
        test_base.TestBase.setUp(self)
        self._bazel_api = BazelApi(lambda args: self.RunBazel(args))
        self.ScratchFile('WORKSPACE')
        self.CreateWorkspaceWithDefaultRepos('repo/WORKSPACE')

    def tearDown(self):
        test_base.TestBase.tearDown(self)

    def testBasicCquery(self):
        self.ScratchFile('testapp/BUILD', [
            'filegroup(name = "fg", srcs = ["a.file"])',
        ])
        (success, stderr, cts)  = self._bazel_api.cquery(["//testapp:all"])
        self.assertTrue(success)
        self.assertEqual(len(cts), 1)
        self.assertEqual(cts[0].label, "//testapp:fg")
        self.assertIsNone(cts[0].config)
        self.assertTrue(len(cts[0].config_hash) > 10)
        self.assertIn("PlatformConfiguration", cts[0].transitive_fragments)

    def testFailedCquery(self):
        self.ScratchFile('testapp/BUILD', [
            'filegroup(name = "fg", srcs = ["a.file"])',
        ])
        (success, stderr, cts)  = self._bazel_api.cquery(["//testapp:typo"])
        self.assertFalse(success)
        self.assertEqual(len(cts), 0)
        self.assertIn(
            "target 'typo' not declared in package 'testapp'",
            os.linesep.join(stderr))

if __name__ == "__main__":
    unittest.main()
