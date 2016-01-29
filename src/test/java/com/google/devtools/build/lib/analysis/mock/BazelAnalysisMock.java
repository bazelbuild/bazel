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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.bazel.rules.BazelConfigurationCollection;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.bazel.rules.python.BazelPythonConfiguration;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.util.BazelMockCcSupport;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JvmConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.ObjcConfigurationLoader;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public final class BazelAnalysisMock extends AnalysisMock {
  public static final AnalysisMock INSTANCE = new BazelAnalysisMock();

  private BazelAnalysisMock() {
  }

  @Override
  public void setupMockClient(MockToolsConfig config) throws IOException {
    String bazelToolWorkspace = config.getPath("/bazel_tools_workspace").getPathString();
    ArrayList<String> workspaceContents =
        new ArrayList<>(
            ImmutableList.of(
                "local_repository(name = 'bazel_tools', path = '" + bazelToolWorkspace + "')",
                "bind(",
                "  name = 'objc_proto_lib',",
                "  actual = '//objcproto:ProtocolBuffers_lib',",
                ")",
                "bind(",
                "  name = 'objc_proto_cpp_lib',",
                "  actual = '//objcproto:ProtocolBuffersCPP_lib',",
                ")",
                "bind(name = 'android/sdk', actual='@bazel_tools//tools/android:sdk')",
                "bind(name = 'tools/python', actual='//tools/python')"));

    config.overwrite("WORKSPACE", workspaceContents.toArray(new String[workspaceContents.size()]));
    config.create("/bazel_tools_workspace/WORKSPACE", "workspace(name = 'bazel_tools')");
    config.create(
        "/bazel_tools_workspace/tools/jdk/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_toolchain(name = 'toolchain', encoding = 'UTF-8', source_version = '8', ",
        "  target_version = '8')",
        "filegroup(name = 'jdk-null')",
        "filegroup(name = 'jdk-default', srcs = [':java'], path = 'jdk/jre')",
        "filegroup(name = 'jdk', srcs = [':jdk-default', ':jdk-null'])",
        "filegroup(name='langtools', srcs=['jdk/lib/tools.jar'])",
        "filegroup(name='bootclasspath', srcs=['jdk/jre/lib/rt.jar'])",
        "filegroup(name='extdir', srcs=glob(['jdk/jre/lib/ext/*']))",
        // "dummy" is needed so that RedirectChaser stops here
        "filegroup(name='java', srcs = ['jdk/jre/bin/java', 'dummy'])",
        "exports_files(['JavaBuilder_deploy.jar','SingleJar_deploy.jar','TestRunner_deploy.jar',",
        "               'JavaBuilderCanary_deploy.jar', 'ijar', 'GenClass_deploy.jar'])");


    ImmutableList<String> androidBuildContents = createAndroidBuildContents();
    config.create(
        "/bazel_tools_workspace/tools/android/BUILD",
        androidBuildContents.toArray(new String[androidBuildContents.size()]));

    config.create(
        "/bazel_tools_workspace/tools/genrule/BUILD", "exports_files(['genrule-setup.sh'])");
    config.create(
        "/bazel_tools_workspace/third_party/java/jarjar/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "java_binary(name = 'jarjar_bin',",
        "            runtime_deps = [ ':jarjar_import' ],",
        "            main_class = 'com.tonicsystems.jarjar.Main')",
        "java_import(name = 'jarjar_import',",
        "            jars = [ 'jarjar.jar' ])");

    config.create("/bazel_tools_workspace/tools/test/BUILD", "filegroup(name = 'runtime')");

    config.create(
        "/bazel_tools_workspace/tools/python/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "exports_files(['precompile.py'])",
        "sh_binary(name='2to3', srcs=['2to3.sh'])");
    ccSupport().setup(config);
  }

  private ImmutableList<String> createAndroidBuildContents() {
    RuleClass androidSdkRuleClass =
        TestRuleClassProvider.getRuleClassProvider().getRuleClassMap().get("android_sdk");

    List<Attribute> attrs = androidSdkRuleClass.getAttributes();
    Builder<String> androidBuildContents = ImmutableList.builder();
    androidBuildContents
        .add("android_sdk(")
        .add("    name = 'sdk',")
        .add("    android_jack = ':empty',")
        .add("    jack = ':fail',")
        .add("    jill = ':fail',")
        .add("    resource_extractor = ':fail',");

    for (Attribute attr : attrs) {
      if (attr.getType() == LABEL && attr.isMandatory() && !attr.getName().startsWith(":")) {
        androidBuildContents.add("    " + attr.getName() + " = ':" + attr.getName() + "',");
      }
    }
    androidBuildContents
        .add(")")
        .add("sh_binary(name = 'aar_generator', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'resources_processor', srcs = ['empty.sh'])")
        .add("android_library(name = 'incremental_stub_application')")
        .add("android_library(name = 'incremental_split_stub_application')")
        .add("sh_binary(name = 'stubify_manifest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'merge_dexzips', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'merge_manifests', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'build_split_manifest', srcs = ['empty.sh'])")
        .add("filegroup(name = 'debug_keystore', srcs = ['fake.file'])")
        .add("sh_binary(name = 'shuffle_jars', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'strip_resources', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'build_incremental_dexmanifest', srcs = ['empty.sh'])")
        .add("sh_binary(name = 'incremental_install', srcs = ['empty.sh'])")
        .add("java_binary(name = 'PackageParser',")
        .add("          runtime_deps = [ ':PackageParser_import'],")
        .add("          main_class = 'com.google.devtools.build.android.ideinfo.PackageParser')")
        .add("java_import(name = 'PackageParser_import',")
        .add("          jars = [ 'package_parser_deploy.jar' ])");

    for (Attribute attr : attrs) {
      if (attr.getType() == LABEL && attr.isMandatory() && !attr.getName().startsWith(":")) {
        if (attr.isExecutable()) {
          androidBuildContents
              .add("sh_binary(name = '" + attr.getName() + "',")
              .add("          srcs = ['empty.sh'],")
              .add(")");
        } else {
          androidBuildContents
              .add("filegroup(name = '" + attr.getName() + "',")
              .add("          srcs = ['fake.file'])");
        }
      }
    }
    androidBuildContents
        .add("java_binary(name = 'IdlClass',")
        .add("            runtime_deps = [ ':idlclass_import' ],")
        .add("            main_class = 'com.google.devtools.build.android.idlclass.IdlClass')")
        .add("java_import(name = 'idlclass_import',")
        .add("            jars = [ 'idlclass.jar' ])");

    return androidBuildContents.build();
  }

  @Override
  public void setupMockWorkspaceFiles(Path embeddedBinariesRoot) throws IOException {
    Path jdkWorkspacePath = embeddedBinariesRoot.getRelative("jdk.WORKSPACE");
    FileSystemUtils.createDirectoryAndParents(jdkWorkspacePath.getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(jdkWorkspacePath, "");
  }

  public static String readFromResources(String filename) {
    try (InputStream in = BazelAnalysisMock.class.getClassLoader().getResourceAsStream(filename)) {
      return new String(ByteStreams.toByteArray(in), StandardCharsets.UTF_8);
    } catch (IOException e) {
      // This should never happen.
      throw new AssertionError(e);
    }
  }

  @Override
  public ConfigurationFactory createConfigurationFactory() {
    return new ConfigurationFactory(new BazelConfigurationCollection(),
        new BazelConfiguration.Loader(),
        new CppConfigurationLoader(Functions.<String>identity()),
        new PythonConfigurationLoader(Functions.<String>identity()),
        new BazelPythonConfiguration.Loader(),
        new JvmConfigurationLoader(false, BazelRuleClassProvider.JAVA_CPU_SUPPLIER),
        new JavaConfigurationLoader(),
        new ObjcConfigurationLoader(),
        new AppleConfiguration.Loader(),
        new J2ObjcConfiguration.Loader(),
        new AndroidConfiguration.Loader());
  }

  @Override
  public ConfigurationCollectionFactory createConfigurationCollectionFactory() {
    return new BazelConfigurationCollection();
  }

  @Override
  public Collection<String> getOptionOverrides() {
    return ImmutableList.of();
  }

  @Override
  public MockCcSupport ccSupport() {
    return BazelMockCcSupport.INSTANCE;
  }
}
