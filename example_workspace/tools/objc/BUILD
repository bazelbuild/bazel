package(default_visibility = ["//visibility:public"])

exports_files(glob(["**"]))

sh_binary(
  name = "bundlemerge",
  srcs = ["bundlemerge.sh"],
  data = [":bundlemerge_deploy.jar"],
)

sh_binary(
  name = "plmerge",
  srcs = ["plmerge.sh"],
  data = [":plmerge_deploy.jar"],
)

sh_binary(
  name = "xcodegen",
  srcs = ["xcodegen.sh"],
  data = [":xcodegen_deploy.jar"],
)
