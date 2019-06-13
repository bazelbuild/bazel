def _impl(ctx):
  fail("Constraints from @bazel_tools//platforms have been removed. " +
       "Please use constraints from @platforms repository embedded in " +
       "Bazel, or preferably declare dependency on " +
       "https://github.com/bazelbuild/platforms. See " +
       "https://github.com/bazelbuild/bazel/issues/8622 for details.")

fail_with_incompatible_use_platforms_repo_for_constraints = rule(
    implementation = _impl,
)
