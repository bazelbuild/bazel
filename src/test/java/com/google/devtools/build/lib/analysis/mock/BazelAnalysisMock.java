// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.analysis.mock;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.io.MoreFiles;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ShellConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.bzlmod.LocalPathOverride;
import com.google.devtools.build.lib.bazel.bzlmod.NonRegistryOverride;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformFunction;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformRule;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.util.BazelMockCcSupport;
import com.google.devtools.build.lib.packages.util.BazelMockPythonSupport;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockGenruleSupport;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import com.google.devtools.build.lib.packages.util.MockPythonSupport;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** Subclass of {@link AnalysisMock} using Bazel-specific semantics. */
public final class BazelAnalysisMock extends AnalysisMock {
  public static final AnalysisMock INSTANCE = new BazelAnalysisMock();

  private BazelAnalysisMock() {}

  @Override
  public ImmutableList<String> getWorkspaceContents(MockToolsConfig config) {
    String xcodeWorkspace = config.getPath("local_config_xcode_workspace").getPathString();
    String protobufWorkspace = config.getPath("protobuf_workspace").getPathString();
    String bazelToolWorkspace = config.getPath("embedded_tools").getPathString();
    String bazelPlatformsWorkspace = config.getPath("platforms_workspace").getPathString();
    String rulesJavaWorkspace = config.getPath("rules_java_workspace").getPathString();
    String androidGmavenR8Workspace = config.getPath("android_gmaven_r8").getPathString();
    String localConfigPlatformWorkspace =
        config.getPath("local_config_platform_workspace").getPathString();
    String appleSupport = config.getPath("build_bazel_apple_support").getPathString();

    return ImmutableList.of(
        "# __SKIP_WORKSPACE_PREFIX__",
        "bind(name = 'android/sdk', actual ="
            + " '@bazel_tools//tools/android:poison_pill_android_sdk')",
        "bind(name = 'android/dx_jar_import', actual ="
            + " '@bazel_tools//tools/android:no_android_sdk_repository_error')",
        "bind(name = 'android/d8_jar_import', actual ="
            + " '@bazel_tools//tools/android:no_android_sdk_repository_error')",
        "bind(name = 'android/crosstool', actual = '@bazel_tools//tools/cpp:toolchain')",
        "bind(name = 'android_sdk_for_testing', actual = '@bazel_tools//tools/android:empty')",
        "bind(name = 'android_ndk_for_testing', actual = '@bazel_tools//tools/android:empty')",
        "bind(name = 'databinding_annotation_processor', actual ="
            + " '@bazel_tools//tools/android:empty')",
        "bind(name = 'has_androidsdk', actual = '@bazel_tools//tools/android:always_false')",
        "local_repository(name = 'bazel_tools', path = '" + bazelToolWorkspace + "')",
        "local_repository(name = 'platforms', path = '" + bazelPlatformsWorkspace + "')",
        "local_repository(name = 'local_config_xcode', path = '" + xcodeWorkspace + "')",
        "local_repository(name = 'com_google_protobuf', path = '" + protobufWorkspace + "')",
        "local_repository(name = 'rules_java', path = '" + rulesJavaWorkspace + "')",
        "local_repository(name = 'rules_java_builtin', path = '" + rulesJavaWorkspace + "')",
        "local_repository(name = 'android_gmaven_r8', path = '" + androidGmavenR8Workspace + "')",
        "local_repository(name = 'build_bazel_apple_support', path = '" + appleSupport + "')",
        "register_toolchains('@rules_java//java/toolchains/runtime:all')",
        "register_toolchains('@rules_java//java/toolchains/javac:all')",
        "bind(name = 'android/sdk', actual='@bazel_tools//tools/android:sdk')",
        "register_toolchains('@bazel_tools//tools/cpp:all')",
        "register_toolchains('@bazel_tools//tools/jdk:all')",
        "register_toolchains('@bazel_tools//tools/android:all')",
        // Note this path is created inside the test infrastructure in
        // createAndroidBuildContents() below. It may not reflect a real depot path.
        "register_toolchains('@bazel_tools//tools/android/dummy_sdk:all')",
        "register_toolchains('@bazel_tools//tools/python:autodetecting_toolchain')",
        "local_repository(name = 'local_config_platform', path = '"
            + localConfigPlatformWorkspace
            + "')");
  }

  /** Keep this in sync with the WORKSPACE content in {@link #getWorkspaceContents}. */
  @Override
  public ImmutableList<String> getWorkspaceRepos() {
    return ImmutableList.of(
        "android_gmaven_r8",
        "bazel_tools",
        "com_google_protobuf",
        "local_config_platform",
        "local_config_xcode",
        "platforms",
        "rules_java",
        "rules_java_builtin",
        "build_bazel_apple_support");
  }

  @Override
  public void setupMockClient(MockToolsConfig config) throws IOException {
    List<String> workspaceContents = getWorkspaceContents(config);
    setupMockClient(config, workspaceContents);
  }

  @Override
  public void setupMockClient(MockToolsConfig config, List<String> workspaceContents)
      throws IOException {
    config.create("local_config_xcode_workspace/WORKSPACE");
    config.create("local_config_xcode_workspace/BUILD", "xcode_config(name = 'host_xcodes')");
    config.create(
        "local_config_xcode_workspace/MODULE.bazel", "module(name = 'local_config_xcode')");
    config.create(
        "protobuf_workspace/BUILD",
        """
        licenses(["notice"])

        exports_files([
            "protoc",
            "cc_toolchain",
        ])
        """);
    config.create("protobuf_workspace/WORKSPACE");
    config.create("protobuf_workspace/MODULE.bazel", "module(name='com_google_protobuf')");
    config.overwrite("WORKSPACE", workspaceContents.toArray(new String[0]));
    config.overwrite("MODULE.bazel");
    /* The rest of platforms is initialized in {@link MockPlatformSupport}. */
    config.create("platforms_workspace/WORKSPACE", "workspace(name = 'platforms')");
    config.create("platforms_workspace/MODULE.bazel", "module(name = 'platforms')");
    config.create(
        "local_config_platform_workspace/WORKSPACE", "workspace(name = 'local_config_platform')");
    config.create(
        "local_config_platform_workspace/MODULE.bazel", "module(name = 'local_config_platform')");
    config.create("build_bazel_apple_support/WORKSPACE", "workspace(name = 'apple_support')");
    config.create(
        "build_bazel_apple_support/MODULE.bazel", "module(name = 'build_bazel_apple_support')");
    config.create("embedded_tools/WORKSPACE", "workspace(name = 'bazel_tools')");
    Runfiles runfiles = Runfiles.create();
    for (String filename :
        Arrays.asList("tools/jdk/java_toolchain_alias.bzl", "tools/jdk/java_stub_template.txt")) {
      java.nio.file.Path path = Paths.get(runfiles.rlocation("io_bazel/" + filename));
      if (!Files.exists(path)) {
        continue; // the io_bazel workspace root only exists for Bazel
      }
      config.create("embedded_tools/" + filename, MoreFiles.asCharSource(path, UTF_8).read());
    }
    config.create(
        "embedded_tools/tools/jdk/launcher_flag_alias.bzl",
        "_providers = [CcInfo, cc_common.launcher_provider]",
        "def _impl(ctx):",
        "    if not ctx.attr._launcher:",
        "      return None",
        "    launcher = ctx.attr._launcher",
        "    providers = [ctx.attr._launcher[p] for p in _providers]",
        "    providers.append(DefaultInfo(files = launcher[DefaultInfo].files, runfiles ="
            + " launcher[DefaultInfo].default_runfiles))",
        "    return providers",
        "launcher_flag_alias = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "        '_launcher': attr.label(",
        "            default = configuration_field(",
        "                fragment = 'java',",
        "                name = 'launcher',",
        "            ),",
        "            providers = _providers,",
        "        ),",
        "    },",
        ")");
    config.create(
        "embedded_tools/tools/jdk/BUILD",
        """
        load(
            ":java_toolchain_alias.bzl",
            "java_host_runtime_alias",
            "java_runtime_alias",
            "java_toolchain_alias",
        )
        load(":launcher_flag_alias.bzl", "launcher_flag_alias")

        package(default_visibility = ["//visibility:public"])

        java_toolchain(
            name = "toolchain",
            bootclasspath = [":bootclasspath"],
            genclass = ["GenClass_deploy.jar"],
            header_compiler = ["turbine_deploy.jar"],
            header_compiler_direct = ["TurbineDirect_deploy.jar"],
            ijar = ["ijar"],
            jacocorunner = ":JacocoCoverage",
            java_runtime = "host_jdk",
            javabuilder = ["JavaBuilder_deploy.jar"],
            singlejar = ["singlejar"],
            source_version = "8",
            target_version = "8",
        )

        java_toolchain(
            name = "remote_toolchain",
            bootclasspath = [":bootclasspath"],
            genclass = ["GenClass_deploy.jar"],
            header_compiler = ["turbine_deploy.jar"],
            header_compiler_direct = ["TurbineDirect_deploy.jar"],
            ijar = ["ijar"],
            jacocorunner = ":JacocoCoverage",
            java_runtime = "host_jdk",
            javabuilder = ["JavaBuilder_deploy.jar"],
            singlejar = ["singlejar"],
            source_version = "8",
            target_version = "8",
        )

        java_import(
            name = "JacocoCoverageRunner",
            jars = ["JacocoCoverage_jarjar_deploy.jar"],
        )

        java_import(
            name = "proguard_import",
            jars = ["proguard_rt.jar"],
        )

        java_binary(
            name = "proguard",
            main_class = "proguard.Proguard",
            runtime_deps = [":proguard_import"],
        )

        java_import(
            name = "TestRunner",
            jars = ["TestRunner.jar"],
        )

        java_runtime(
            name = "jdk",
            srcs = [],
        )

        java_runtime(
            name = "host_jdk",
            srcs = [],
        )

        java_runtime(
            name = "remote_jdk11",
            srcs = [],
        )

        java_toolchain_alias(name = "current_java_toolchain")

        java_runtime_alias(name = "current_java_runtime")

        java_host_runtime_alias(name = "current_host_java_runtime")

        filegroup(
            name = "bootclasspath",
            srcs = ["jdk/jre/lib/rt.jar"],
        )

        filegroup(
            name = "extdir",
            srcs = glob(
                ["jdk/jre/lib/ext/*"],
                allow_empty = True,
            ),
        )

        filegroup(
            name = "java",
            srcs = ["jdk/jre/bin/java"],
        )

        filegroup(
            name = "JacocoCoverage",
            srcs = ["JacocoCoverage_deploy.jar"],
        )

        exports_files([
            "JavaBuilder_deploy.jar",
            "singlejar",
            "TestRunner_deploy.jar",
            "ijar",
            "GenClass_deploy.jar",
            "turbine_deploy.jar",
            "TurbineDirect_deploy.jar",
            "proguard_allowlister.par",
        ])

        toolchain_type(name = "toolchain_type")

        toolchain_type(name = "runtime_toolchain_type")

        toolchain(
            name = "dummy_java_toolchain",
            toolchain = ":toolchain",
            toolchain_type = ":toolchain_type",
        )

        toolchain(
            name = "dummy_java_runtime_toolchain",
            toolchain = ":jdk",
            toolchain_type = ":runtime_toolchain_type",
        )

        java_plugins_flag_alias(name = "java_plugins_flag_alias")

        launcher_flag_alias(
            name = "launcher_flag_alias",
            visibility = ["//visibility:public"],
        )
        """);

    config.create(
        TestConstants.CONSTRAINTS_PATH + "/android/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "platform(",
        "  name = 'armeabi-v7a',",
        "  parents = ['" + TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT + ":host'],",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:android',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7',",
        "  ],",
        ")");

    // Create the actual SDKs.
    ImmutableList<String> androidBuildContents = createAndroidBuildContents();
    config.create(
        "embedded_tools/tools/android/BUILD", androidBuildContents.toArray(new String[0]));
    config.create(
        "embedded_tools/src/tools/android/java/com/google/devtools/build/android/r8/BUILD",
        "java_library(name='r8')\n");
    config.create(
        "embedded_tools/tools/android/emulator/BUILD",
        Iterables.toArray(createToolsAndroidEmulatorContents(), String.class));
    config.create(
        "embedded_tools/tools/android/dummy_sdk/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        toolchain(
            name = "dummy-sdk",
            toolchain = ":invalid-fallback-sdk",
            toolchain_type = "@bazel_tools//tools/android:sdk_toolchain_type",
        )

        filegroup(
            name = "jar-filegroup",
            srcs = ["dummy.jar"],
        )

        genrule(
            name = "empty-binary",
            srcs = [],
            outs = ["empty.sh"],
            cmd = "touch $@",
            executable = 1,
        )

        android_sdk(
            name = "invalid-fallback-sdk",
            aapt = ":empty_binary",
            aapt2 = ":empty_binary",
            adb = ":empty_binary",
            aidl = ":empty_binary",
            android_jar = ":jar-filegroup",
            apksigner = ":empty_binary",
            dx = ":empty_binary",
            framework_aidl = "dummy.jar",
            main_dex_classes = "dummy.jar",
            main_dex_list_creator = ":empty_binary",
            proguard = "empty_binary",
            shrinked_android_jar = "dummy.jar",
            tags = ["__ANDROID_RULES_MIGRATION__"],
            zipalign = ":empty_binary",
        )
        """);
    config.create(
        "android_gmaven_r8/jar/BUILD",
        """
        java_import(
            name = "jar",
            jars = ["r8.jar"],
        )

        filegroup(
            name = "file",
            srcs = [],
        )
        """);
    config.create("android_gmaven_r8/WORKSPACE");

    MockGenruleSupport.setup(config);

    config.create(
        "embedded_tools/tools/test/BUILD",
        """
        filegroup(
            name = "runtime",
            srcs = [
                "test-setup.sh",
                "test-xml-generator.sh",
            ],
        )

        filegroup(
            name = "test_wrapper",
            srcs = ["test_wrapper_bin"],
        )

        filegroup(
            name = "xml_writer",
            srcs = ["xml_writer_bin"],
        )

        filegroup(
            name = "test_setup",
            srcs = ["test-setup.sh"],
        )

        filegroup(
            name = "test_xml_generator",
            srcs = ["test-xml-generator.sh"],
        )

        filegroup(
            name = "collect_coverage",
            srcs = ["collect_coverage.sh"],
        )

        filegroup(
            name = "collect_cc_coverage",
            srcs = ["collect_cc_coverage.sh"],
        )

        filegroup(
            name = "coverage_support",
            srcs = ["collect_coverage.sh"],
        )

        filegroup(
            name = "coverage_report_generator",
            srcs = ["coverage_report_generator.sh"],
        )

        filegroup(
            name = "lcov_merger",
            srcs = ["lcov_merger.sh"],
        )
        """);

    // Use an alias package group to allow for modification at the simpler path
    config.create(
        "embedded_tools/tools/allowlists/config_feature_flag/BUILD",
        """
        package_group(
            name = "config_feature_flag",
            includes = ["@@//tools/allowlists/config_feature_flag"],
        )

        package_group(
            name = "config_feature_flag_setter",
            includes = ["@@//tools/allowlists/config_feature_flag:config_feature_flag_setter"],
        )
        """);

    config.create(
        "tools/allowlists/config_feature_flag/BUILD",
        """
        package_group(
            name = "config_feature_flag",
            packages = ["public"],
        )

        package_group(
            name = "config_feature_flag_Setter",
            packages = ["public"],
        )
        """);
    config.create(
        "embedded_tools/tools/allowlists/extend_rule_allowlist/BUILD",
        """
        package_group(
            name = "extend_rule_allowlist",
            packages = ["public"],
        )
        """);

    config.create(
        "embedded_tools/tools/allowlists/android_binary_allowlist/BUILD",
        """
package_group(
    name = "enable_starlark_dex_desugar_proguard",
    includes = ["@@//tools/allowlists/android_binary_allowlist:enable_starlark_dex_desugar_proguard"],
)
""");
    config.create(
        "tools/allowlists/android_binary_allowlist/BUILD",
        "package_group(name='enable_starlark_dex_desugar_proguard', packages=[])");

    config.create(
        "embedded_tools/tools/proto/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        alias(
            name = "protoc",
            actual = "@com_google_protobuf//:protoc",
        )

        alias(
            name = "javalite_toolchain",
            actual = "@com_google_protobuf//:javalite_toolchain",
        )

        alias(
            name = "java_toolchain",
            actual = "@com_google_protobuf//:java_toolchain",
        )

        alias(
            name = "cc_toolchain",
            actual = "@com_google_protobuf//:cc_toolchain",
        )
        """);

    config.create(
        "embedded_tools/tools/zip/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        exports_files(["precompile.py"])

        cc_binary(
            name = "zipper",
            srcs = ["zip_main.cc"],
        )

        alias(
            name = "unzip_fdo",
            actual = ":zipper",
        )
        """);

    config.create(
        "embedded_tools/tools/launcher/BUILD",
        """
        load("@bazel_tools//third_party/cc_rules/macros:defs.bzl", "cc_binary")

        package(default_visibility = ["//visibility:public"])

        cc_binary(
            name = "launcher",
            srcs = ["launcher_main.cc"],
        )

        cc_binary(
            name = "launcher_maker",
            srcs = ["launcher_maker.cc"],
        )
        """);

    config.create(
        "embedded_tools/tools/def_parser/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        filegroup(
            name = "def_parser",
            srcs = ["def_parser.exe"],
        )
        """);

    config.create(
        "embedded_tools/objcproto/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        objc_library(
            name = "protobuf_lib",
            srcs = ["empty.m"],
            hdrs = ["include/header.h"],
            includes = ["include"],
        )

        exports_files(["well_known_type.proto"])

        proto_library(
            name = "well_known_type_proto",
            srcs = ["well_known_type.proto"],
        )
        """);
    config.create("embedded_tools/objcproto/empty.m");
    config.create("embedded_tools/objcproto/empty.cc");
    config.create("embedded_tools/objcproto/well_known_type.proto");

    config.create("rules_java_workspace/WORKSPACE", "workspace(name = 'rules_java')");
    config.create("rules_java_workspace/MODULE.bazel", "module(name = 'rules_java')");
    config.create("rules_java_workspace/java/BUILD");
    config.create("rules_java_workspace/toolchains/BUILD");
    java.nio.file.Path path =
        Paths.get(runfiles.rlocation("rules_java/toolchains/java_toolchain_alias.bzl"));
    if (Files.exists(path)) {
      config.create(
          "rules_java_workspace/toolchains/java_toolchain_alias.bzl",
          MoreFiles.asCharSource(path, UTF_8).read());
    }
    config.create(
        "rules_java_workspace/toolchains/local_java_repository.bzl",
        """
        def local_java_repository(**attrs):
            pass
        """);
    config.create("rules_java_workspace/toolchains/jdk_build_file.bzl", "JDK_BUILD_TEMPLATE = ''");
    config.create(
        "rules_java_workspace/java/defs.bzl",
        """
        def java_binary(**attrs):
            native.java_binary(**attrs)

        def java_library(**attrs):
            native.java_library(**attrs)

        def java_import(**attrs):
            native.java_import(**attrs)
        """);
    config.create(
        "rules_java_workspace/java/repositories.bzl",
        """
        def rules_java_dependencies():
            pass

        def rules_java_toolchains():
            native.register_toolchains("//java/toolchains/runtime:all")
            native.register_toolchains("//java/toolchains/javac:all")
        """);

    config.create(
        "rules_java_workspace/java/toolchains/runtime/BUILD",
        """
        toolchain_type(name = "toolchain_type")

        toolchain(
            name = "local_jdk",
            toolchain = "@bazel_tools//tools/jdk:jdk",
            toolchain_type = "@rules_java//java/toolchains/runtime:toolchain_type",
        )
        """);
    config.create(
        "rules_java_workspace/java/toolchains/javac/BUILD",
        """
        toolchain_type(name = "toolchain_type")

        toolchain(
            name = "javac_toolchain",
            toolchain = "@bazel_tools//tools/jdk:toolchain",
            toolchain_type = "@rules_java//java/toolchains/javac:toolchain_type",
        )
        """);

    config.create("third_party/bazel_rules/rules_proto/WORKSPACE");
    config.create("third_party/bazel_rules/rules_proto/MODULE.bazel", "module(name='rules_proto')");

    config.create("third_party/bazel_rules/rules_cc/WORKSPACE");
    config.create("third_party/bazel_rules/rules_cc/MODULE.bazel", "module(name='rules_cc')");

    config.create(
        "embedded_tools/tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = ["public"],
        )
        """);

    MockPlatformSupport.setup(config);
    ccSupport().setup(config);
    pySupport().setup(config);
    ShellConfiguration.injectShellExecutableFinder(
        BazelRuleClassProvider::getDefaultPathFromOptions, BazelRuleClassProvider.SHELL_EXECUTABLE);
  }

  /** Contents of {@code //tools/android/emulator/BUILD.tools}. */
  private ImmutableList<String> createToolsAndroidEmulatorContents() {
    return ImmutableList.of(
        "exports_files(['emulator_arm', 'emulator_x86', 'mksd', 'empty_snapshot_fs'])",
        "filegroup(name = 'emulator_x86_bios', srcs = ['bios.bin', 'vgabios-cirrus.bin'])",
        "filegroup(name = 'xvfb_support', srcs = ['support_file1', 'support_file2'])",
        "sh_binary(name = 'unified_launcher', srcs = ['empty.sh'])",
        "filegroup(name = 'shbase', srcs = ['googletest.sh'])",
        "filegroup(name = 'sdk_path', srcs = ['empty.sh'])");
  }

  private ImmutableList<String> createAndroidBuildContents() {
    ImmutableList.Builder<String> androidBuildContents = ImmutableList.builder();

    androidBuildContents.add(
        "package(default_visibility=['//visibility:public'])",
        "toolchain_type(name = 'sdk_toolchain_type')",
        "toolchain(",
        "  name = 'sdk_toolchain',",
        "  toolchain = ':sdk',",
        "  toolchain_type = ':sdk_toolchain_type',",
        "  target_compatible_with = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:android',",
        "  ],",
        ")",
        "android_sdk(",
        "    name = 'sdk',",
        "    aapt = ':static_aapt_tool',",
        "    aapt2 = ':static_aapt2_tool',",
        "    adb = ':static_adb_tool',",
        "    aidl = ':static_aidl_tool',",
        "    android_jar = ':android_runtime_jar',",
        "    apksigner = ':ApkSignerBinary',",
        "    dx = ':dx_binary',",
        "    framework_aidl = ':aidl_framework',",
        "    main_dex_classes = ':mainDexClasses.rules',",
        "    main_dex_list_creator = ':main_dex_list_creator',",
        "    proguard = ':ProGuard',",
        "    shrinked_android_jar = ':shrinkedAndroid.jar',",
        "    zipalign = ':zipalign',",
        "    tags = ['__ANDROID_RULES_MIGRATION__'],",
        ")",
        "filegroup(name = 'android_runtime_jar', srcs = ['android.jar'])",
        "filegroup(name = 'dx_binary', srcs = ['dx_binary.jar'])");

    androidBuildContents
        .add("sh_binary(name = 'aar_generator', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'desugar_java8', srcs = ['empty.sh'])")
        .add("filegroup(name = 'desugar_java8_extra_bootclasspath', srcs = ['fake.jar'])")
        .add("filegroup(name = 'java8_legacy_dex', srcs = ['java8_legacy.dex.zip'])")
        .add("sh_binary(name = 'build_java8_legacy_dex', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'merge_proguard_maps', srcs = ['empty.sh'])")
        .add("filegroup(name = 'desugared_java8_legacy_apis', srcs = ['fake.jar'])")
        .add("sh_binary(name = 'aar_native_libs_zip_creator', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'resource_extractor', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'dexbuilder', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'dexbuilder_after_proguard', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'dexmerger', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'dexsharder', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'aar_import_deps_checker', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'busybox', srcs = ['empty.sh'])")
        .add("android_library(name = 'incremental_stub_application')")
        .add("android_library(name = 'incremental_split_stub_application')")
        .add("sh_binary(name = 'stubify_manifest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'merge_dexzips', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'build_split_manifest', srcs = ['empty.sh'])")
        .add("filegroup(name = 'debug_keystore', srcs = ['fake.file'])")
        .add("sh_binary(name = 'shuffle_jars', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'strip_resources', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'build_incremental_dexmanifest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'incremental_install', srcs = ['empty.sh'])")
        .add("java_binary(name = 'IdlClass',")
        .add("            runtime_deps = [ ':idlclass_import' ],")
        .add("            main_class = 'com.google.devtools.build.android.idlclass.IdlClass')")
        .add("java_binary(name = 'zip_filter',")
        .add("            main_class = 'com.google.devtools.build.android.ZipFilterAction',")
        .add("            runtime_deps = [ ':ZipFilterAction_import' ])")
        .add("java_import(name = 'ZipFilterAction_import',")
        .add("            jars = [ 'ZipFilterAction_deploy.jar' ])")
        .add("sh_binary(name = 'aar_resources_extractor', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'aar_embedded_jars_extractor', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'aar_embedded_proguard_extractor', srcs = ['empty.sh'])")
        .add("java_import(name = 'idlclass_import',")
        .add("            jars = [ 'idlclass.jar' ])")
        .add("exports_files(['adb', 'adb_static'])")
        .add("sh_binary(name = 'android_runtest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'instrumentation_test_entry_point', srcs = ['empty.sh'])")
        .add("java_plugin(name = 'databinding_annotation_processor',")
        .add("    generates_api = 1,")
        .add("    processor_class = 'android.databinding.annotationprocessor.ProcessDataBinding')")
        .add("sh_binary(name = 'instrumentation_test_check', srcs = ['empty.sh'])")
        .add("package_group(name = 'android_device_allowlist', packages = ['public'])")
        .add("package_group(name = 'export_deps_allowlist', packages = ['public'])")
        .add("package_group(name = 'allow_android_library_deps_without_srcs_allowlist',")
        .add("    packages=['public'])")
        .add("android_tools_defaults_jar(name = 'android_jar')")
        .add("sh_binary(name = 'dex_list_obfuscator', srcs = ['empty.sh'])");

    return androidBuildContents.build();
  }

  @Override
  public void setupMockWorkspaceFiles(Path embeddedBinariesRoot) throws IOException {
    embeddedBinariesRoot.createDirectoryAndParents();
    Path jdkWorkspacePath = embeddedBinariesRoot.getRelative("jdk.WORKSPACE");
    FileSystemUtils.writeContentAsLatin1(jdkWorkspacePath, "");
  }

  @Override
  public void setupMockToolsRepository(MockToolsConfig config) throws IOException {
    config.create("embedded_tools/WORKSPACE", "workspace(name = 'bazel_tools')");
    config.create("embedded_tools/MODULE.bazel", "module(name='bazel_tools')");
    config.create("embedded_tools/tools/build_defs/repo/BUILD");
    config.create(
        "embedded_tools/tools/build_defs/build_info/bazel_cc_build_info.bzl",
        """
        def _impl(ctx):
            volatile_file = ctx.actions.declare_file("volatile_file.h")
            non_volatile_file = ctx.actions.declare_file("non_volatile_file.h")
            redacted_file = ctx.actions.declare_file("redacted_file.h")
            ctx.actions.write(output = volatile_file, content = "")
            ctx.actions.write(output = non_volatile_file, content = "")
            ctx.actions.write(output = redacted_file, content = "")
            output_groups = {
                "non_redacted_build_info_files": depset([volatile_file, non_volatile_file]),
                "redacted_build_info_files": depset([redacted_file]),
            }
            return OutputGroupInfo(**output_groups)

        bazel_cc_build_info = rule(implementation = _impl)
        """);
    config.create(
        "embedded_tools/tools/build_defs/build_info/bazel_java_build_info.bzl",
        """
        def _impl(ctx):
            volatile_file = ctx.actions.declare_file("volatile_file.properties")
            non_volatile_file = ctx.actions.declare_file("non_volatile_file.properties")
            redacted_file = ctx.actions.declare_file("redacted_file.properties")
            ctx.actions.write(output = volatile_file, content = "")
            ctx.actions.write(output = non_volatile_file, content = "")
            ctx.actions.write(output = redacted_file, content = "")
            output_groups = {
                "non_redacted_build_info_files": depset([volatile_file, non_volatile_file]),
                "redacted_build_info_files": depset([redacted_file]),
            }
            return OutputGroupInfo(**output_groups)

        bazel_java_build_info = rule(implementation = _impl)
        """);
    config.create(
        "embedded_tools/tools/build_defs/build_info/BUILD",
        "load('//tools/build_defs/build_info:bazel_cc_build_info.bzl'," + " 'bazel_cc_build_info')",
        "load('//tools/build_defs/build_info:bazel_java_build_info.bzl',"
            + " 'bazel_java_build_info')",
        "bazel_cc_build_info(",
        "    name = 'cc_build_info',",
        "    visibility = ['//visibility:public'],",
        ")",
        "bazel_java_build_info(",
        "    name = 'java_build_info',",
        "    visibility = ['//visibility:public'],",
        ")");
    config.create(
        "embedded_tools/tools/build_defs/repo/utils.bzl",
        """
        def maybe(repo_rule, name, **kwargs):
            if name not in native.existing_rules():
                repo_rule(name = name, **kwargs)
        """);
    config.create(
        "embedded_tools/tools/build_defs/repo/http.bzl",
        """
        def http_archive(**kwargs):
            pass

        def http_file(**kwargs):
            pass

        def http_jar(**kwargs):
            pass
        """);
    config.create("embedded_tools/tools/jdk/jdk_build_file.bzl", "JDK_BUILD_TEMPLATE = ''");
    config.create(
        "embedded_tools/tools/jdk/local_java_repository.bzl",
        """
        def local_java_repository(**kwargs):
            pass
        """);
    config.create(
        "embedded_tools/tools/jdk/remote_java_repository.bzl",
        """
        def remote_java_repository(**kwargs):
            pass
        """);
    config.create(
        "embedded_tools/tools/cpp/cc_configure.bzl",
        """
        def cc_configure(**kwargs):
            pass
        """);

    config.create("embedded_tools/tools/sh/BUILD");
    config.create(
        "embedded_tools/tools/sh/sh_configure.bzl",
        """
        def sh_configure(**kwargs):
            pass
        """);
    config.create("embedded_tools/tools/osx/BUILD");
    config.create(
        "embedded_tools/tools/osx/xcode_configure.bzl",
        """
        # no positional arguments for XCode
        def xcode_configure(*args, **kwargs):
            pass
        """);
    config.create("embedded_tools/bin/sh", "def sh(**kwargs):", "  pass");
  }

  @Override
  public ImmutableMap<String, NonRegistryOverride> getBuiltinModules(BlazeDirectories directories) {
    ImmutableMap<String, String> moduleNameToPath =
        ImmutableMap.<String, String>builder()
            .put("bazel_tools", "embedded_tools")
            .put("platforms", "platforms_workspace")
            .put("local_config_platform", "local_config_platform_workspace")
            .put("rules_java", "rules_java_workspace")
            .put("com_google_protobuf", "protobuf_workspace")
            .put("rules_proto", "third_party/bazel_rules/rules_proto")
            .put("build_bazel_apple_support", "build_bazel_apple_support")
            .put("local_config_xcode", "local_config_xcode_workspace")
            .put("rules_cc", "third_party/bazel_rules/rules_cc")
            .buildOrThrow();
    return moduleNameToPath.entrySet().stream()
        .collect(
            toImmutableMap(
                Map.Entry::getKey,
                e ->
                    LocalPathOverride.create(
                        directories
                            .getWorkingDirectory()
                            .getRelative(e.getValue())
                            .getPathString())));
  }

  @Override
  public ConfiguredRuleClassProvider createRuleClassProvider() {
    return TestRuleClassProvider.getRuleClassProviderWithClearedSuffix();
  }

  @Override
  public boolean isThisBazel() {
    return true;
  }

  @Override
  public MockCcSupport ccSupport() {
    return BazelMockCcSupport.INSTANCE;
  }

  @Override
  public MockPythonSupport pySupport() {
    return BazelMockPythonSupport.INSTANCE;
  }

  @Override
  public void addExtraRepositoryFunctions(
      ImmutableMap.Builder<String, RepositoryFunction> repositoryHandlers) {
    repositoryHandlers.put(LocalConfigPlatformRule.NAME, new LocalConfigPlatformFunction());
  }
}
