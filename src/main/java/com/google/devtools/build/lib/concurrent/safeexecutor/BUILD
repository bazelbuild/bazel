load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__subpackages__"],
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
    visibility = ["//src/main/java/com/google/devtools/build/lib:build_lib"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:concurrent",
    ],
)
