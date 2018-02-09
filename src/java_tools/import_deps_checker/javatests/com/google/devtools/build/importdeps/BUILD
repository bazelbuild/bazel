# Description:
#   Tests for the checker to check the completeness of the deps of java_import or aar_import targets.
package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

java_test(
    name = "ClassInfoTest",
    srcs = ["ClassInfoTest.java"],
    deps = [
        "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:import_deps_checker",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ResultCollectorTest",
    srcs = ["ResultCollectorTest.java"],
    deps = [
        "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:import_deps_checker",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "LazyClassEntryStateTest",
    srcs = ["LazyClassEntryStateTest.java"],
    deps = [
        "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:import_deps_checker",
        "//src/test/java/com/google/devtools/build/lib/testutil:JunitUtils",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "ClassCacheTest",
    srcs = [
        "AbstractClassCacheTest.java",
        "ClassCacheTest.java",
    ],
    data = [
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_client",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryAnnotations",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryException",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryInterface",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library_no_members",
        "//third_party/java/jdk:jdk8_rt_jar",
    ],
    jvm_flags = [
        "-Dclasscache.test.bootclasspath=$(location //third_party/java/jdk:jdk8_rt_jar)",
        "-Dclasscache.test.Client=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_client)",
        "-Dclasscache.test.Library=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library)",
        "-Dclasscache.test.Library_no_members=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library_no_members)",
        "-Dclasscache.test.LibraryAnnotations=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryAnnotations)",
        "-Dclasscache.test.LibraryException=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryException)",
        "-Dclasscache.test.LibraryInterface=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryInterface)",
    ],
    test_class = "com.google.devtools.build.importdeps.ClassCacheTest",
    deps = [
        "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:import_deps_checker",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "DepsCheckerClassVisitorTest",
    srcs = [
        "AbstractClassCacheTest.java",
        "DepsCheckerClassVisitorTest.java",
    ],
    data = [
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_client",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryAnnotations",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryException",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryInterface",
        "//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library_no_members",
        "//third_party/java/jdk:jdk8_rt_jar",
    ],
    jvm_flags = [
        "-Dclasscache.test.bootclasspath=$(location //third_party/java/jdk:jdk8_rt_jar)",
        "-Dclasscache.test.Client=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_client)",
        "-Dclasscache.test.Library=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library)",
        "-Dclasscache.test.Library_no_members=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_Library_no_members)",
        "-Dclasscache.test.LibraryAnnotations=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryAnnotations)",
        "-Dclasscache.test.LibraryException=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryException)",
        "-Dclasscache.test.LibraryInterface=$(location //src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps/testdata:testdata_lib_LibraryInterface)",
    ],
    test_class = "com.google.devtools.build.importdeps.DepsCheckerClassVisitorTest",
    deps = [
        "//src/java_tools/import_deps_checker/java/com/google/devtools/build/importdeps:import_deps_checker",
        "//third_party:asm",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
        "//third_party/java/jdk:jdk8_rt_jar",
    ],
)

load("//src/java_tools/import_deps_checker/javatests/com/google/devtools/build/importdeps:tests.bzl", "create_golden_test")

create_golden_test(
    name = "bootclasspath_missing_golden_test",
    golden_file = "golden_bootclasspath_missing.txt",
    has_bootclasspath = False,
    missing_jar = None,
)

create_golden_test(
    name = "library_exception_missing_golden_test",
    golden_file = "golden_library_exception_missing.txt",
    has_bootclasspath = True,
    missing_jar = "testdata_lib_LibraryException",
)

create_golden_test(
    name = "library_annotation_missing_golden_test",
    golden_file = "golden_library_annotation_missing.txt",
    has_bootclasspath = True,
    missing_jar = "testdata_lib_LibraryAnnotations",
)

create_golden_test(
    name = "library_missing_golden_test",
    golden_file = "golden_library_missing.txt",
    has_bootclasspath = True,
    missing_jar = "testdata_lib_Library",
)

create_golden_test(
    name = "library_interface_missing_golden_test",
    golden_file = "golden_library_interface_missing.txt",
    has_bootclasspath = True,
    missing_jar = "testdata_lib_LibraryInterface",
)

create_golden_test(
    name = "library_members_missing_golden_test",
    golden_file = "golden_library_members_missing.txt",
    has_bootclasspath = True,
    missing_jar = "testdata_lib_Library",
    replacing_jar = "testdata_lib_Library_no_members",
)

create_golden_test(
    name = "complete_classpath_golden_test",
    golden_file = "golden_complete_classpath.txt",
    has_bootclasspath = True,
    missing_jar = None,
    replacing_jar = None,
)
