Test Data for Bazel Integration Tests
=====================================

`git_repository_test.sh`
------------------------

The following archives contain test Git repositories used by
`git_repository_test.sh` to test the `git_repository` and `new_git_repository`
workspace rules:

* outer-planets-repo.tar.gz
* pluto-repo.tar.gz
* refetch-repo.tar.gz

For reference, the following files contain the output of `git log -p --decorate`
for the two repositories:

* outer-planets.git_log
* pluto.git_log
* refetch.git_log

These files were created by manually creating a git repository and tarring up
the result using `tar -zcvf`.
