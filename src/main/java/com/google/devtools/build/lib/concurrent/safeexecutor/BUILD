load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "safe_executor",
    srcs = [
        "RejectionHandlingRunnable.java",
        "SafeCallbackListener.java",
        "SafeDirectExecutor.java",
        "SafeExecutor.java",
        "SafeExecutorOwner.java",
        "SafeFutures.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)
