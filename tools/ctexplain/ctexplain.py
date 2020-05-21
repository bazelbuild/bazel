# Lint as: python3

from tools.ctexplain.bazel_api import BazelApi

bazel_api = BazelApi()
print("Ready to go.")
k = bazel_api.cquery(["//tools/ctexplain/examples/simple:split"])
print("status code: " + str(k[0]))
print("stderr: " + str(k[1], "utf-8"))
print("=" * 30)
print("results: " + str(k[2]))

print("==" * 30)
print(f"Let's get the config for {k[2][0].config_hash}:")
config = bazel_api.get_config(k[2][0].config_hash)
print(config)
