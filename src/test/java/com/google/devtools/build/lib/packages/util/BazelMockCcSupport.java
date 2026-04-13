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
package com.google.devtools.build.lib.packages.util;

import static java.lang.Integer.MAX_VALUE;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.Arrays;

/** Bazel implementation of {@link MockCcSupport} */
public final class BazelMockCcSupport extends MockCcSupport {
  public static final BazelMockCcSupport INSTANCE = new BazelMockCcSupport();

  /** Filter to remove implicit dependencies of C/C++ rules. */
  private static final boolean isNotCcLabel(String label) {
    return !label.startsWith("//tools/cpp");
  }

  private BazelMockCcSupport() {}

  private static final ImmutableList<String> CROSSTOOL_ARCHS =
      ImmutableList.of("piii", "k8", "armeabi-v7a", "ppc", "darwin_x86_64");

  @Override
  protected String getRealFilesystemCrosstoolTopPath() {
    if (OS.getCurrent() == OS.LINUX) {
      return "src/test/java/com/google/devtools/build/lib/packages/util/real/linux";
    }
    throw new IllegalStateException("Unsupported OS: " + OS.getCurrent());
  }

  @Override
  protected String[] getRealFilesystemToolsToLink(String crosstoolTop) {
    return new String[0];
  }

  @Override
  protected String[] getRealFilesystemToolsToCopy(String crosstoolTop) {
    return new String[] {crosstoolTop + "/BUILD"};
  }

  @Override
  protected ImmutableList<String> getCrosstoolArchs() {
    return CROSSTOOL_ARCHS;
  }

  @Override
  public void setup(MockToolsConfig config) throws IOException {
    writeMacroFile(config);
    setupRulesCc(config);
    setupCcToolchainConfig(config, getToolchainConfigs());
    createParseHeadersAndLayeringCheckWhitelist(config);
    createStarlarkLooseHeadersWhitelist(config, "//...");
    config.append(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "alias(name='host_xcodes',actual='@local_config_xcode//:host_xcodes')");
    if (config.isRealFileSystem() && shouldUseRealFileSystemCrosstool()) {
      config.append(
          TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/BUILD",
          """
          toolchain_type(name = 'toolchain_type')
          cc_library(
              name = 'link_extra_lib',
              srcs = ['linkextra.cc'],
              tags = ['__DONT_DEPEND_ON_DEF_PARSER__'],
          )
          cc_library(
              name = 'malloc',
              srcs = ['malloc.cc'],
              tags = ['__DONT_DEPEND_ON_DEF_PARSER__'],
          )
          filegroup(
              name = 'aggregate-ddi',
              srcs = ['aggregate-ddi.sh'],
          )
          filegroup(
              name = 'generate-modmap',
              srcs = ['generate-modmap.sh'],
          )
          filegroup(
              name = 'interface_library_builder',
              srcs = ['interface_library_builder.sh'],
          )
          filegroup(
              name = 'link_dynamic_library',
              srcs = ['link_dynamic_library.sh'],
          )
          """);
      for (String s :
          Arrays.asList(
              "linkextra.cc",
              "malloc.cc",
              "aggregate-ddi.sh",
              "generate-modmap.sh",
              "interface_library_builder.sh",
              "link_dynamic_library.sh")) {
        config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "tools/cpp/" + s);
      }
    }

    // Copies rules_cc from real @rules_cc
    config.create("third_party/bazel_rules/rules_cc/MODULE.bazel", "module(name='rules_cc')");
    Runfiles runfiles = Runfiles.preload().withSourceRepository("");
    PathFragment path = PathFragment.create(runfiles.rlocation("rules_cc/cc/defs.bzl"));
    config.copyDirectory(
        path.getParentDirectory(), "third_party/bazel_rules/rules_cc/cc", MAX_VALUE, true);

    // avoid cc_compatibility_proxy indirection
    for (String ruleName :
        ImmutableList.of(
            "cc_binary",
            "cc_import",
            "cc_library",
            "cc_shared_library",
            "cc_static_library",
            "cc_test",
            "objc_import",
            "objc_library")) {
      config.overwrite(
          "third_party/bazel_rules/rules_cc/cc/" + ruleName + ".bzl",
          MessageFormat.format(
              """
              load("//cc/private/rules_impl:{0}.bzl", _{0} = "{0}")
              {0} = _{0}
              """,
              ruleName));
    }
    for (String ruleName : ImmutableList.of("cc_toolchain", "cc_toolchain_alias")) {
      config.overwrite(
          "third_party/bazel_rules/rules_cc/cc/toolchains/" + ruleName + ".bzl",
          MessageFormat.format(
              """
              load("//cc/private/rules_impl:{0}.bzl", _{0} = "{0}")
              {0} = _{0}
              """,
              ruleName));
    }
    for (String ruleName :
        ImmutableList.of("fdo_prefetch_hints", "fdo_profile", "propeller_optimize")) {
      config.overwrite(
          "third_party/bazel_rules/rules_cc/cc/toolchains/" + ruleName + ".bzl",
          MessageFormat.format(
              """
              load("//cc/private/rules_impl/fdo:{0}.bzl", _{0} = "{0}")
              {0} = _{0}
              """,
              ruleName));
    }
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/common/cc_info.bzl",
        """
        load("//cc/private:cc_info.bzl", _CcInfo = "CcInfo")
        CcInfo = _CcInfo
        """);
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/common/cc_shared_library_info.bzl",
        """
        load("//cc/private:cc_shared_library_info.bzl", _CcSharedLibraryInfo = "CcSharedLibraryInfo")
        CcSharedLibraryInfo = _CcSharedLibraryInfo
        """);
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/common/debug_package_info.bzl",
        """
        load("//cc/private:debug_package_info.bzl", _DebugPackageInfo = "DebugPackageInfo")
        DebugPackageInfo = _DebugPackageInfo
        """);
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/common/cc_common.bzl",
        """
        load("//cc/private:cc_common.bzl", _cc_common = "cc_common")
        cc_common = _cc_common
        """);
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/common/objc_info.bzl",
        """
        load("//cc/private:objc_info.bzl", _ObjcInfo = "ObjcInfo")
        ObjcInfo = _ObjcInfo
        """);
    config.overwrite(
        "third_party/bazel_rules/rules_cc/cc/toolchains/cc_toolchain_config_info.bzl",
        """
        load("//cc/private/toolchain_config:cc_toolchain_config_info.bzl", _CcToolchainConfigInfo = "CcToolchainConfigInfo")
        CcToolchainConfigInfo = _CcToolchainConfigInfo
        """);
    config.overwrite("third_party/bazel_rules/rules_cc/cc/toolchains/BUILD");
    config.overwrite("third_party/bazel_rules/rules_cc/cc/toolchains/impl/BUILD");
    config.overwrite("third_party/bazel_rules/rules_cc/cc/common/BUILD");
    config.overwrite("third_party/bazel_rules/rules_cc/cc/private/BUILD");
  }

  @Override
  public Label getMockCrosstoolLabel() {
    return Label.parseCanonicalUnchecked("@bazel_tools//tools/cpp:toolchain");
  }

  @Override
  public String getMockCrosstoolPath() {
    return "embedded_tools/tools/cpp/";
  }

  @Override
  public Predicate<String> labelNameFilter() {
    return BazelMockCcSupport::isNotCcLabel;
  }

  @Override
  protected boolean shouldUseRealFileSystemCrosstool() {
    return OS.getCurrent() == OS.LINUX;
  }

  private static ImmutableList<CcToolchainConfig> getToolchainConfigs() {
    ImmutableList.Builder<CcToolchainConfig> result = ImmutableList.builder();

    // Different from CcToolchainConfig.getDefault....
    result.add(CcToolchainConfig.builder().build());

    if (OS.getCurrent() == OS.DARWIN) {
      result.add(CcToolchainConfig.getCcToolchainConfigForCpu("darwin_x86_64"));
      result.add(CcToolchainConfig.getCcToolchainConfigForCpu("darwin_arm64"));
    }

    if (System.getProperty("os.arch").equals("s390x")) {
      result.add(CcToolchainConfig.getCcToolchainConfigForCpu("s390x"));
    }
    return result.build();
  }
}
