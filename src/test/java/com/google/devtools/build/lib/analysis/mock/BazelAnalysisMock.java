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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.PlatformConfigurationLoader;
import com.google.devtools.build.lib.analysis.ShellConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformFunction;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformRule;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider.StrictActionEnvOptions;
import com.google.devtools.build.lib.bazel.rules.python.BazelPythonConfiguration;
import com.google.devtools.build.lib.packages.util.BazelMockCcSupport;
import com.google.devtools.build.lib.packages.util.BazelMockPythonSupport;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockPythonSupport;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.swift.SwiftConfiguration;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.cpp.CpuTransformer;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcConfigurationLoader;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Subclass of {@link AnalysisMock} using Bazel-specific semantics. */
public final class BazelAnalysisMock extends AnalysisMock {
  public static final AnalysisMock INSTANCE = new BazelAnalysisMock();

  private BazelAnalysisMock() {
  }

  @Override
  public List<String> getWorkspaceContents(MockToolsConfig config) {
    String bazelToolWorkspace = config.getPath("/bazel_tools_workspace").getPathString();
    String localConfigPlatformWorkspace =
        config.getPath("/local_config_platform_workspace").getPathString();
    return new ArrayList<>(
        ImmutableList.of(
            "local_repository(name = 'bazel_tools', path = '" + bazelToolWorkspace + "')",
            "local_repository(name = 'local_config_xcode', path = '/local_config_xcode')",
            "local_repository(name = 'com_google_protobuf', path = '/protobuf')",
            "bind(name = 'android/sdk', actual='@bazel_tools//tools/android:sdk')",
            "register_toolchains('@bazel_tools//tools/cpp:all')",
            "register_toolchains('@bazel_tools//tools/jdk:dummy_java_toolchain')",
            "register_toolchains('@bazel_tools//tools/jdk:dummy_java_runtime_toolchain')",
            "local_repository(name = 'local_config_platform', path = '"
                + localConfigPlatformWorkspace
                + "')"));
  }

  @Override
  public void setupMockClient(MockToolsConfig config) throws IOException {
    List<String> workspaceContents = getWorkspaceContents(config);
    setupMockClient(config, workspaceContents);
  }

  @Override
  public void setupMockClient(MockToolsConfig config, List<String> workspaceContents)
      throws IOException {
    config.create("/local_config_xcode/BUILD", "xcode_config(name = 'host_xcodes')");
    config.create(
        "/protobuf/BUILD", "licenses(['notice'])", "exports_files(['protoc', 'cc_toolchain'])");
    config.create("/local_config_xcode/WORKSPACE");
    config.create("/protobuf/WORKSPACE");
    config.overwrite("WORKSPACE", workspaceContents.toArray(new String[workspaceContents.size()]));
    config.create("/bazel_tools_workspace/WORKSPACE", "workspace(name = 'bazel_tools')");
    config.create(
        "/bazel_tools_workspace/tools/jdk/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_toolchain(",
        "  name = 'toolchain',",
        "  source_version = '8',",
        "  target_version = '8',",
        "  bootclasspath = [':bootclasspath'],",
        "  extclasspath = [':extclasspath'],",
        "  javac = [':langtools'],",
        "  javabuilder = ['JavaBuilder_deploy.jar'],",
        "  header_compiler = ['turbine_deploy.jar'],",
        "  singlejar = ['SingleJar_deploy.jar'],",
        "  genclass = ['GenClass_deploy.jar'],",
        "  ijar = ['ijar'],",
        ")",
        "java_runtime(name = 'jdk', srcs = [])",
        "java_runtime(name = 'host_jdk', srcs = [])",
        "java_runtime(name = 'remote_jdk', srcs = [])",
        "java_runtime(name = 'remote_jdk10', srcs = [])",
        "java_runtime(name = 'remote_jdk11', srcs = [])",
        "java_runtime_alias(name = 'current_java_runtime')",
        "java_host_runtime_alias(name = 'current_host_java_runtime')",
        "filegroup(name='langtools', srcs=['jdk/lib/tools.jar'])",
        "filegroup(name='bootclasspath', srcs=['jdk/jre/lib/rt.jar'])",
        "filegroup(name='extdir', srcs=glob(['jdk/jre/lib/ext/*']))",
        "filegroup(name='java', srcs = ['jdk/jre/bin/java'])",
        "filegroup(name='JacocoCoverage', srcs = [])",
        "filegroup(name='jacoco-blaze-agent', srcs = [])",
        "exports_files(['JavaBuilder_deploy.jar','SingleJar_deploy.jar','TestRunner_deploy.jar',",
        "               'JavaBuilderCanary_deploy.jar', 'ijar', 'GenClass_deploy.jar',",
        "               'turbine_deploy.jar','ExperimentalTestRunner_deploy.jar'])",
        "sh_binary(name = 'proguard_whitelister', srcs = ['empty.sh'])",
        "toolchain_type(name = 'toolchain_type')",
        "toolchain_type(name = 'runtime_toolchain_type')",
        "toolchain(",
        "   name = 'dummy_java_toolchain',",
        "   toolchain_type = ':toolchain_type',",
        "   toolchain = ':toolchain',",
        ")",
        "toolchain(",
        "   name = 'dummy_java_runtime_toolchain',",
        "   toolchain_type = ':runtime_toolchain_type',",
        "   toolchain = ':jdk',",
        ")");

    ImmutableList<String> androidBuildContents = createAndroidBuildContents();
    config.create(
        "/bazel_tools_workspace/tools/android/BUILD",
        androidBuildContents.toArray(new String[androidBuildContents.size()]));
    config.create(
        "/bazel_tools_workspace/tools/android/emulator/BUILD",
        Iterables.toArray(createToolsAndroidEmulatorContents(), String.class));
    // Bundled Proguard used by android_sdk_repository
    config.create(
        "/bazel_tools_workspace/third_party/java/proguard/BUILD",
        "exports_files(['proguard'])");

    config.create(
        "/bazel_tools_workspace/tools/genrule/BUILD", "exports_files(['genrule-setup.sh'])");

    config.create(
        "/bazel_tools_workspace/tools/test/BUILD",
        "filegroup(name = 'runtime', srcs = ['test-setup.sh', 'test-xml-generator.sh'])",
        "filegroup(name = 'test_wrapper', srcs = ['test_wrapper_bin'])",
        "filegroup(name = 'xml_writer', srcs = ['xml_writer_bin'])",
        "filegroup(name = 'test_setup', srcs = ['test-setup.sh'])",
        "filegroup(name = 'test_xml_generator', srcs = ['test-xml-generator.sh'])",
        "filegroup(name = 'collect_coverage', srcs = ['collect_coverage.sh'])",
        "filegroup(name = 'collect_cc_coverage', srcs = ['collect_cc_coverage.sh'])",
        "filegroup(name='coverage_support', srcs=['collect_coverage.sh'])",
        "filegroup(name = 'coverage_report_generator', srcs = ['coverage_report_generator.sh'])");

    config.create(
        "/bazel_tools_workspace/tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator/BUILD",
        "filegroup(name='srcs', srcs = glob(['**']))",
        "filegroup(name='Main', srcs = ['Main.java'])");

    // Use an alias package group to allow for modification at the simpler path
    config.create(
        "/bazel_tools_workspace/tools/whitelists/config_feature_flag/BUILD",
        "package_group(",
        "    name='config_feature_flag',",
        "    includes=['@//tools/whitelists/config_feature_flag'],",
        ")");

    config.create(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(name='config_feature_flag', packages=['//...'])");

    config.create(
        "tools/whitelists/config_feature_flag/BUILD",
        "package_group(name='config_feature_flag', packages=['//...'])");

    config.create(
        "/bazel_tools_workspace/tools/zip/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "exports_files(['precompile.py'])",
        "cc_binary(name='zipper', srcs=['zip_main.cc'])");

    config.create(
        "/bazel_tools_workspace/tools/launcher/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_binary(name='launcher', srcs=['launcher_main.cc'])");

    config.create(
        "/bazel_tools_workspace/tools/def_parser/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "filegroup(name='def_parser', srcs=['def_parser.exe'])");

    config.create(
        "/bazel_tools_workspace/objcproto/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "objc_library(",
        "  name = 'protobuf_lib',",
        "  srcs = ['empty.m'],",
        "  hdrs = ['include/header.h'],",
        "  includes = ['include'],",
        ")",
        "exports_files(['well_known_type.proto'])",
        "proto_library(",
        "  name = 'well_known_type_proto',",
        "  srcs = ['well_known_type.proto'],",
        ")");
    config.create("/bazel_tools_workspace/objcproto/empty.m");
    config.create("/bazel_tools_workspace/objcproto/empty.cc");
    config.create("/bazel_tools_workspace/objcproto/well_known_type.proto");

    ccSupport().setup(config);
    pySupport().setup(config);
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
        "android_sdk(",
        "    name = 'sdk',",
        "    aapt = ':static_aapt_tool',",
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
        .add("java_import(name = 'idlclass_import',")
        .add("            jars = [ 'idlclass.jar' ])")
        .add("exports_files(['adb', 'adb_static'])")
        .add("sh_binary(name = 'android_runtest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'instrumentation_test_entry_point', srcs = ['empty.sh'])")
        .add("java_plugin(name = 'databinding_annotation_processor',")
        .add("    generates_api = 1,")
        .add("    processor_class = 'android.databinding.annotationprocessor.ProcessDataBinding')")
        .add("sh_binary(name = 'jarjar_bin', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'instrumentation_test_check', srcs = ['empty.sh'])")
        .add("package_group(name = 'android_device_whitelist', packages = ['//...'])")
        .add("package_group(name = 'export_deps_whitelist', packages = ['//...'])")
        .add("package_group(name = 'allow_android_library_deps_without_srcs_whitelist',")
        .add("    packages=['//...'])")
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
  public List<ConfigurationFragmentFactory> getDefaultConfigurationFragmentFactories() {
    return ImmutableList.of(
        new CppConfigurationLoader(CpuTransformer.IDENTITY),
        new ShellConfiguration.Loader(
            BazelRuleClassProvider.SHELL_EXECUTABLE,
            ShellConfiguration.Options.class,
            StrictActionEnvOptions.class),
        new PythonConfigurationLoader(),
        new BazelPythonConfiguration.Loader(),
        new JavaConfigurationLoader(),
        new ObjcConfigurationLoader(),
        new AppleConfiguration.Loader(),
        new SwiftConfiguration.Loader(),
        new J2ObjcConfiguration.Loader(),
        new ProtoConfiguration.Loader(),
        new ConfigFeatureFlagConfiguration.Loader(),
        new AndroidConfiguration.Loader(),
        new PlatformConfigurationLoader());
  }

  @Override
  public ConfiguredRuleClassProvider createRuleClassProvider() {
    return TestRuleClassProvider.getRuleClassProvider(true);
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
