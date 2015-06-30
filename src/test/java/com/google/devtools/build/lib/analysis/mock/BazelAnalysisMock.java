// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.bazel.rules.BazelConfigurationCollection;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JvmConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.ObjcConfigurationLoader;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;

public class BazelAnalysisMock extends AnalysisMock {
  public static final AnalysisMock INSTANCE = new BazelAnalysisMock();

  @Override
  public void setupMockClient(MockToolsConfig config) throws IOException {
    config.create("WORKSPACE");
    config.create("tools/defaults/BUILD");
    config.create("tools/jdk/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_toolchain(name = 'toolchain', encoding = 'UTF-8', source_version = '8', ",
        "  target_version = '8')",
        "filegroup(name = 'jdk-null')",
        "filegroup(name = 'jdk-default', srcs = [':java'], path = 'jdk/jre')",
        "filegroup(name = 'jdk', srcs = [':jdk-default', ':jdk-null'])",
        "filegroup(name='langtools', srcs=['jdk/lib/tools.jar'])",
        "filegroup(name='bootclasspath', srcs=['jdk/jre/lib/rt.jar'])",
        "filegroup(name='extdir', srcs=glob(['jdk/jre/lib/ext/*']))",
        "filegroup(name='java', srcs = ['jdk/jre/bin/java'])",
        "exports_files(['JavaBuilder_deploy.jar','SingleJar_deploy.jar',",
        "               'JavaBuilderCanary_deploy.jar', 'ijar'])");
    config.create("tools/cpp/BUILD",
        "filegroup(name = 'toolchain', srcs = [':cc-compiler-local', ':empty'])",
        "cc_toolchain(name = 'cc-compiler-k8', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-piii', all_files = ':empty', compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")",
        "cc_toolchain(name = 'cc-compiler-darwin', all_files = ':empty', ",
        "    compiler_files = ':empty',",
        "    cpu = 'local', dwp_files = ':empty', dynamic_runtime_libs = [':empty'], ",
        "    linker_files = ':empty',",
        "    objcopy_files = ':empty', static_runtime_libs = [':empty'], strip_files = ':empty',",
        ")");
    config.create("tools/cpp/CROSSTOOL", readFromResources("MOCK_CROSSTOOL"));
    config.create("tools/android/BUILD",
        "filegroup(name = 'sdk')",
        "filegroup(name = 'aar_generator')",
        "filegroup(name = 'resources_processor')",
        "filegroup(name = 'incremental_stub_application')",
        "filegroup(name = 'incremental_split_stub_application')");
    config.create("tools/genrule/BUILD",
        "exports_files(['genrule-setup.sh'])");
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
        new JvmConfigurationLoader(false, BazelRuleClassProvider.JAVA_CPU_SUPPLIER),
        new JavaConfigurationLoader(),
        new ObjcConfigurationLoader());
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
  public ImmutableList<Class<? extends FragmentOptions>> getBuildOptions() {
    return BazelRuleClassProvider.BUILD_OPTIONS;
  }
}
