TEST_SUITES = [
    "rules",
    "analysis",
    "foundations",
    "shell",
    "skyframe",
    "ideinfo",
]

[
    test_suite(
        name = n,
        tests = ["//src/test/java/com/google/devtools/build/lib:" + n],
    )
    for n in TEST_SUITES
]

test_suite(
    name = "ziputils",
    tests = ["//src/test/java/com/google/devtools/build/android/ziputils:ziputils-tests"],
)

test_suite(
    name = "others",
    tests = [
        "//src/test/java/com/google/devtools/build/lib:others",
        "//src/test/java/com/google/devtools/common/options:options_test",
    ],
)
