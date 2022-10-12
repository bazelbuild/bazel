"""Give the path to the package rewriter. 

For Bazel itself, this only adds package metadata. Other organizations can do
different things.
"""

repo_patcher_label = Label("@bazel_module_patcher//:add_package_metadata.py")
