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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link CppConfigurationLoader}.
 */
@RunWith(JUnit4.class)
public class CrosstoolConfigurationLoaderTest extends AnalysisTestCase {

  private CppConfiguration create(CppConfigurationLoader loader, String... args) throws Exception {
    useConfiguration(args);
    ConfigurationEnvironment env =
        new ConfigurationEnvironment.TargetProviderEnvironment(
            skyframeExecutor.getPackageManager(), reporter);
    return loader.create(env, buildOptions);
  }

  private CppConfigurationLoader loader(String crosstoolFileContents) throws IOException {
    getAnalysisMock().ccSupport().setupCrosstoolWithRelease(mockToolsConfig, crosstoolFileContents);
    return new CppConfigurationLoader(CpuTransformer.IDENTITY);
  }

  @Before
  public void setupTests() throws Exception {
    useRuleClassProvider(TestRuleClassProvider.getRuleClassProvider());
  }

  private CppConfigurationLoader loaderWithOptionalTool(String optionalTool) throws IOException {
    return loader(
        "major_version: \"12\""
            + "minor_version: \"0\""
            + "toolchain {"
            + "  toolchain_identifier: \"toolchain-identifier\""
            + "  host_system_name: \"host-system-name\""
            + "  target_system_name: \"target-system-name\""
            + "  target_cpu: \"k8\""
            + "  target_libc: \"target-libc\""
            + "  compiler: \"compiler\""
            + "  abi_version: \"abi-version\""
            + "  abi_libc_version: \"abi-libc-version\""
            + "  tool_path { name: \"ar\" path: \"path-to-ar\" }"
            + "  tool_path { name: \"cpp\" path: \"path-to-cpp\" }"
            + "  tool_path { name: \"gcc\" path: \"path-to-gcc\" }"
            + "  tool_path { name: \"gcov\" path: \"path-to-gcov\" }"
            + "  tool_path { name: \"ld\" path: \"path-to-ld\" }"
            + "  tool_path { name: \"nm\" path: \"path-to-nm\" }"
            + "  tool_path { name: \"objcopy\" path: \"path-to-objcopy\" }"
            + "  tool_path { name: \"objdump\" path: \"path-to-objdump\" }"
            + "  tool_path { name: \"strip\" path: \"path-to-strip\" }"
            + "  tool_path { name: \"dwp\" path: \"path-to-dwp\" }"
            + optionalTool
            + "  supports_gold_linker: true"
            + "  supports_normalizing_ar: true"
            + "  supports_incremental_linker: true"
            + "  supports_fission: true"
            + "  compiler_flag: \"c\""
            + "  cxx_flag: \"cxx\""
            + "  unfiltered_cxx_flag: \"unfiltered\""
            + "  linker_flag: \"linker\""
            + "  dynamic_library_linker_flag: \"solinker\""
            + "  objcopy_embed_flag: \"objcopy\""
            + "  compilation_mode_flags {"
            + "    mode: FASTBUILD"
            + "    compiler_flag: \"fastbuild\""
            + "    cxx_flag: \"cxx-fastbuild\""
            + "    linker_flag: \"linker-fastbuild\""
            + "  }"
            + "  compilation_mode_flags {"
            + "    mode: DBG"
            + "    compiler_flag: \"dbg\""
            + "    cxx_flag: \"cxx-dbg\""
            + "    linker_flag: \"linker-dbg\""
            + "  }"
            + "  compilation_mode_flags {"
            + "    mode: COVERAGE"
            + "    compiler_flag: \"coverage\""
            + "    cxx_flag: \"cxx-coverage\""
            + "    linker_flag: \"linker-coverage\""
            + "  }"
            + "  compilation_mode_flags {"
            + "    mode: OPT"
            + "    compiler_flag: \"opt\""
            + "    cxx_flag: \"cxx-opt\""
            + "    linker_flag: \"linker-opt\""
            + "  }"
            + "  linking_mode_flags {"
            + "    mode: FULLY_STATIC"
            + "    linker_flag: \"fully static\""
            + "  }"
            + "  linking_mode_flags {"
            + "    mode: MOSTLY_STATIC"
            + "    linker_flag: \"mostly static\""
            + "  }"
            + "  linking_mode_flags {"
            + "    mode: DYNAMIC"
            + "    linker_flag: \"dynamic\""
            + "  }"
            + "  make_variable {"
            + "    name: \"SOME_MAKE_VARIABLE\""
            + "    value: \"make-variable-value\""
            + "  }"
            + "  cxx_builtin_include_directory: \"system-include-dir\""
            + "}");
  }

  private ConfiguredTarget getCcToolchainTarget(CppConfiguration cppConfiguration)
      throws Exception {
    update(cppConfiguration.getRuleProvidingCcToolchainProvider().toString());
    return Preconditions.checkNotNull(
        getConfiguredTarget(cppConfiguration.getRuleProvidingCcToolchainProvider().toString()));
  }

  private CcToolchainProvider getCcToolchainProvider(CppConfiguration cppConfiguration)
      throws Exception {
    return (CcToolchainProvider)
        getCcToolchainTarget(cppConfiguration).get(ToolchainInfo.PROVIDER);
  }

  /**
   * Checks that we do not accidentally change the proto format in incompatible
   * ways. Do not modify the configuration file in this test, except if you are
   * absolutely certain that it is backwards-compatible.
   */
  @Test
  public void testSimpleCompleteConfiguration() throws Exception {
    CppConfigurationLoader loader = loaderWithOptionalTool("");

    // Need to clear out the android cpu options to avoid this split transition in Bazel.
    CppConfiguration toolchain =
        create(loader, "--cpu=k8", "--host_cpu=k8", "--android_cpu=", "--fat_apk_cpu=");
    CcToolchainProvider ccProvider = getCcToolchainProvider(toolchain);
    assertThat(ccProvider.getToolchainIdentifier()).isEqualTo("toolchain-identifier");

    assertThat(ccProvider.getHostSystemName()).isEqualTo("host-system-name");
    assertThat(ccProvider.getCompiler()).isEqualTo("compiler");
    assertThat(ccProvider.getTargetLibc()).isEqualTo("target-libc");
    assertThat(ccProvider.getTargetCpu()).isEqualTo("k8");
    assertThat(ccProvider.getTargetGnuSystemName()).isEqualTo("target-system-name");

    assertThat(ccProvider.getToolPathFragment(Tool.AR)).isEqualTo(getToolPath("path-to-ar"));

    assertThat(ccProvider.getAbi()).isEqualTo("abi-version");
    assertThat(ccProvider.getAbiGlibcVersion()).isEqualTo("abi-libc-version");

    assertThat(ccProvider.supportsGoldLinker()).isTrue();
    assertThat(ccProvider.supportsStartEndLib()).isFalse();
    assertThat(ccProvider.supportsInterfaceSharedObjects()).isFalse();
    assertThat(ccProvider.supportsEmbeddedRuntimes()).isFalse();
    assertThat(ccProvider.toolchainNeedsPic()).isFalse();
    assertThat(ccProvider.supportsFission()).isTrue();

    assertThat(ccProvider.getBuiltInIncludeDirectories())
        .containsExactly(getToolPath("system-include-dir"));
    assertThat(ccProvider.getSysroot()).isNull();

    assertThat(ccProvider.getLegacyCompileOptionsWithCopts())
        .containsExactly("c", "fastbuild")
        .inOrder();
    assertThat(ccProvider.getUnfilteredCompilerOptions()).containsExactly("unfiltered").inOrder();

    assertThat(CppHelper.getFullyStaticLinkOptions(toolchain, ccProvider, false))
        .containsExactly("linker", "linker-fastbuild", "fully static")
        .inOrder();
    assertThat(CppHelper.getDynamicLinkOptions(toolchain, ccProvider, false))
        .containsExactly("linker", "linker-fastbuild", "dynamic")
        .inOrder();
    assertThat(CppHelper.getFullyStaticLinkOptions(toolchain, ccProvider, true))
        .containsExactly("linker", "linker-fastbuild", "mostly static", "solinker")
        .inOrder();
    assertThat(CppHelper.getDynamicLinkOptions(toolchain, ccProvider, true))
        .containsExactly("linker", "linker-fastbuild", "dynamic", "solinker")
        .inOrder();

    assertThat(ccProvider.getObjCopyOptionsForEmbedding()).containsExactly("objcopy").inOrder();
    assertThat(ccProvider.getLdOptionsForEmbedding()).isEmpty();

    assertThat(ccProvider.getAdditionalMakeVariables().entrySet())
        .containsExactlyElementsIn(
            ImmutableMap.of(
                    "SOME_MAKE_VARIABLE", "make-variable-value",
                    "STACK_FRAME_UNLIMITED", "",
                    "CC_FLAGS", "")
                .entrySet());

    assertThat(ccProvider.getToolPathFragment(Tool.LD)).isEqualTo(getToolPath("path-to-ld"));
    assertThat(ccProvider.getToolPathFragment(Tool.DWP)).isEqualTo(getToolPath("path-to-dwp"));
  }

  /**
   * Tests all of the fields and a bunch of the combinations a config can hold,
   * including non-default toolchains, missing sections and repeated entries
   * (and their order in the end result.)
   */
  @Test
  public void testComprehensiveCompleteConfiguration() throws Exception {
    CppConfigurationLoader loader =
        loader(
            // Needs to include \n's; as a single line it hits a parser limitation.
            "major_version: \"12\"\n"
                + "minor_version: \"0\"\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-A\"\n"
                + "  host_system_name: \"host-system-name-A\"\n"
                + "  target_system_name: \"target-system-name-A\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "  tool_path { name: \"ar\" path: \"path/to/ar-A\" }\n"
                + "  tool_path { name: \"cpp\" path: \"path/to/cpp-A\" }\n"
                + "  tool_path { name: \"gcc\" path: \"path/to/gcc-A\" }\n"
                + "  tool_path { name: \"gcov\" path: \"path/to/gcov-A\" }\n"
                + "  tool_path { name: \"gcov-tool\" path: \"path-to-gcov-tool-A\" }"
                + "  tool_path { name: \"ld\" path: \"path/to/ld-A\" }\n"
                + "  tool_path { name: \"nm\" path: \"path/to/nm-A\" }\n"
                + "  tool_path { name: \"objcopy\" path: \"path/to/objcopy-A\" }\n"
                + "  tool_path { name: \"objdump\" path: \"path/to/objdump-A\" }\n"
                + "  tool_path { name: \"strip\" path: \"path/to/strip-A\" }\n"
                + "  tool_path { name: \"dwp\" path: \"path/to/dwp\" }\n"
                + "  supports_gold_linker: true\n"
                + "  supports_start_end_lib: true\n"
                + "  supports_normalizing_ar: true\n"
                + "  supports_embedded_runtimes: true\n"
                + "  needsPic: true\n"
                + "  compiler_flag: \"compiler-flag-A-1\"\n"
                + "  compiler_flag: \"compiler-flag-A-2\"\n"
                + "  cxx_flag: \"cxx-flag-A-1\"\n"
                + "  cxx_flag: \"cxx-flag-A-2\"\n"
                + "  unfiltered_cxx_flag: \"unfiltered-flag-A-1\"\n"
                + "  unfiltered_cxx_flag: \"unfiltered-flag-A-2\"\n"
                + "  linker_flag: \"linker-flag-A-1\"\n"
                + "  linker_flag: \"linker-flag-A-2\"\n"
                + "  dynamic_library_linker_flag: \"solinker-flag-A-1\"\n"
                + "  dynamic_library_linker_flag: \"solinker-flag-A-2\"\n"
                + "  objcopy_embed_flag: \"objcopy-embed-flag-A-1\"\n"
                + "  objcopy_embed_flag: \"objcopy-embed-flag-A-2\"\n"
                + "  ld_embed_flag: \"ld-embed-flag-A-1\"\n"
                + "  ld_embed_flag: \"ld-embed-flag-A-2\"\n"
                + "  compilation_mode_flags {\n"
                + "    mode: FASTBUILD\n"
                + "    compiler_flag: \"fastbuild-flag-A-1\"\n"
                + "    compiler_flag: \"fastbuild-flag-A-2\"\n"
                + "    cxx_flag: \"cxx-fastbuild-flag-A-1\"\n"
                + "    cxx_flag: \"cxx-fastbuild-flag-A-2\"\n"
                + "    linker_flag: \"linker-fastbuild-flag-A-1\"\n"
                + "    linker_flag: \"linker-fastbuild-flag-A-2\"\n"
                + "  }\n"
                + "  compilation_mode_flags {\n"
                + "    mode: DBG\n"
                + "    compiler_flag: \"dbg-flag-A-1\"\n"
                + "    compiler_flag: \"dbg-flag-A-2\"\n"
                + "    cxx_flag: \"cxx-dbg-flag-A-1\"\n"
                + "    cxx_flag: \"cxx-dbg-flag-A-2\"\n"
                + "    linker_flag: \"linker-dbg-flag-A-1\"\n"
                + "    linker_flag: \"linker-dbg-flag-A-2\"\n"
                + "  }\n"
                + "  compilation_mode_flags {\n"
                + "    mode: COVERAGE\n"
                + "  }\n"
                + "  # skip mode OPT to test handling its absence\n"
                + "  linking_mode_flags {\n"
                + "    mode: FULLY_STATIC\n"
                + "    linker_flag: \"fully-static-flag-A-1\"\n"
                + "    linker_flag: \"fully-static-flag-A-2\"\n"
                + "  }\n"
                + "  linking_mode_flags {\n"
                + "    mode: MOSTLY_STATIC\n"
                + "  }\n"
                + "  # skip linking mode DYNAMIC to test handling its absence\n"
                + "  make_variable {\n"
                + "    name: \"SOME_MAKE_VARIABLE-A-1\"\n"
                + "    value: \"make-variable-value-A-1\"\n"
                + "  }\n"
                + "  make_variable {\n"
                + "    name: \"SOME_MAKE_VARIABLE-A-2\"\n"
                + "    value: \"make-variable-value-A-2 with spaces in\"\n"
                + "  }\n"
                + "  cxx_builtin_include_directory: \"system-include-dir-A-1\"\n"
                + "  cxx_builtin_include_directory: \"system-include-dir-A-2\"\n"
                + "  builtin_sysroot: \"builtin-sysroot-A\"\n"
                + "  default_python_top: \"python-top-A\"\n"
                + "  default_python_version: \"python-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-B\"\n"
                + "  host_system_name: \"host-system-name-B\"\n"
                + "  target_system_name: \"target-system-name-B\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-B\"\n"
                + "  compiler: \"compiler-B\"\n"
                + "  abi_version: \"abi-version-B\"\n"
                + "  abi_libc_version: \"abi-libc-version-B\"\n"
                + "  tool_path { name: \"ar\" path: \"path/to/ar-B\" }\n"
                + "  tool_path { name: \"cpp\" path: \"path/to/cpp-B\" }\n"
                + "  tool_path { name: \"gcc\" path: \"path/to/gcc-B\" }\n"
                + "  tool_path { name: \"gcov\" path: \"path/to/gcov-B\" }\n"
                + "  tool_path { name: \"gcov-tool\" path: \"path/to/gcov-tool-B\" }\n"
                + "  tool_path { name: \"ld\" path: \"path/to/ld-B\" }\n"
                + "  tool_path { name: \"nm\" path: \"path/to/nm-B\" }\n"
                + "  tool_path { name: \"objcopy\" path: \"path/to/objcopy-B\" }\n"
                + "  tool_path { name: \"objdump\" path: \"path/to/objdump-B\" }\n"
                + "  tool_path { name: \"strip\" path: \"path/to/strip-B\" }\n"
                + "  tool_path { name: \"dwp\" path: \"path/to/dwp\" }\n"
                + "  supports_gold_linker: true\n"
                + "  supports_start_end_lib: true\n"
                + "  supports_normalizing_ar: true\n"
                + "  supports_embedded_runtimes: true\n"
                + "  needsPic: true\n"
                + "  compiler_flag: \"compiler-flag-B-1\"\n"
                + "  compiler_flag: \"compiler-flag-B-2\"\n"
                + "  cxx_flag: \"cxx-flag-B-1\"\n"
                + "  cxx_flag: \"cxx-flag-B-2\"\n"
                + "  unfiltered_cxx_flag: \"unfiltered-flag-B-1\"\n"
                + "  unfiltered_cxx_flag: \"unfiltered-flag-B-2\"\n"
                + "  linker_flag: \"linker-flag-B-1\"\n"
                + "  linker_flag: \"linker-flag-B-2\"\n"
                + "  dynamic_library_linker_flag: \"solinker-flag-B-1\"\n"
                + "  dynamic_library_linker_flag: \"solinker-flag-B-2\"\n"
                + "  objcopy_embed_flag: \"objcopy-embed-flag-B-1\"\n"
                + "  objcopy_embed_flag: \"objcopy-embed-flag-B-2\"\n"
                + "  ld_embed_flag: \"ld-embed-flag-B-1\"\n"
                + "  ld_embed_flag: \"ld-embed-flag-B-2\"\n"
                + "  compilation_mode_flags {\n"
                + "    mode: FASTBUILD\n"
                + "    compiler_flag: \"fastbuild-flag-B-1\"\n"
                + "    compiler_flag: \"fastbuild-flag-B-2\"\n"
                + "    cxx_flag: \"cxx-fastbuild-flag-B-1\"\n"
                + "    cxx_flag: \"cxx-fastbuild-flag-B-2\"\n"
                + "    linker_flag: \"linker-fastbuild-flag-B-1\"\n"
                + "    linker_flag: \"linker-fastbuild-flag-B-2\"\n"
                + "  }\n"
                + "  compilation_mode_flags {\n"
                + "    mode: DBG\n"
                + "    compiler_flag: \"dbg-flag-B-1\"\n"
                + "    compiler_flag: \"dbg-flag-B-2\"\n"
                + "    cxx_flag: \"cxx-dbg-flag-B-1\"\n"
                + "    cxx_flag: \"cxx-dbg-flag-B-2\"\n"
                + "    linker_flag: \"linker-dbg-flag-B-1\"\n"
                + "    linker_flag: \"linker-dbg-flag-B-2\"\n"
                + "  }\n"
                + "  compilation_mode_flags {\n"
                + "    mode: COVERAGE\n"
                + "  }\n"
                + "  # skip mode OPT to test handling its absence\n"
                + "  linking_mode_flags {\n"
                + "    mode: FULLY_STATIC\n"
                + "    linker_flag: \"fully-static-flag-B-1\"\n"
                + "    linker_flag: \"fully-static-flag-B-2\"\n"
                + "  }\n"
                + "  linking_mode_flags {\n"
                + "    mode: MOSTLY_STATIC\n"
                + "  }\n"
                + "  # skip linking mode DYNAMIC to test handling its absence\n"
                + "  make_variable {\n"
                + "    name: \"SOME_MAKE_VARIABLE-B-1\"\n"
                + "    value: \"make-variable-value-B-1\"\n"
                + "  }\n"
                + "  make_variable {\n"
                + "    name: \"SOME_MAKE_VARIABLE-B-2\"\n"
                + "    value: \"make-variable-value-B-2 with spaces in\"\n"
                + "  }\n"
                + "  cxx_builtin_include_directory: \"system-include-dir-B-1\"\n"
                + "  cxx_builtin_include_directory: \"system-include-dir-B-2\"\n"
                + "  builtin_sysroot: \"builtin-sysroot-B\"\n"
                + "  default_python_top: \"python-top-B\"\n"
                + "  default_python_version: \"python-version-B\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-C\"\n"
                + "  host_system_name: \"host-system-name-C\"\n"
                + "  target_system_name: \"target-system-name-C\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-C\"\n"
                + "  compiler: \"compiler-C\"\n"
                + "  abi_version: \"abi-version-C\"\n"
                + "  abi_libc_version: \"abi-libc-version-C\"\n"
                + "  tool_path { name: \"ar\" path: \"path/to/ar-C\" }"
                + "  tool_path { name: \"cpp\" path: \"path/to/cpp-C\" }"
                + "  tool_path { name: \"gcc\" path: \"path/to/gcc-C\" }"
                + "  tool_path { name: \"gcov\" path: \"path/to/gcov-C\" }"
                + "  tool_path { name: \"gcov-tool\" path: \"path/to/gcov-tool-C\" }"
                + "  tool_path { name: \"ld\" path: \"path/to/ld-C\" }"
                + "  tool_path { name: \"nm\" path: \"path/to/nm-C\" }"
                + "  tool_path { name: \"objcopy\" path: \"path/to/objcopy-C\" }"
                + "  tool_path { name: \"objdump\" path: \"path/to/objdump-C\" }"
                + "  tool_path { name: \"strip\" path: \"path/to/strip-C\" }"
                + "  tool_path { name: \"dwp\" path: \"path/to/dwp\" }\n"
                + "}");

    mockToolsConfig.create(
        "some/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['unencumbered'])",
        "filegroup(name = 'everything')");

    // Need to clear out the android cpu options to avoid this split transition in Bazel.
    CppConfiguration toolchainA =
        create(loader, "--cpu=k8", "--host_cpu=k8", "--android_cpu=", "--fat_apk_cpu=");
    ConfiguredTarget ccToolchainA = getCcToolchainTarget(toolchainA);
    CcToolchainProvider ccProviderA =
        (CcToolchainProvider) ccToolchainA.get(ToolchainInfo.PROVIDER);
    TemplateVariableInfo makeProviderA = ccToolchainA.get(TemplateVariableInfo.PROVIDER);
    assertThat(ccProviderA.getToolchainIdentifier()).isEqualTo("toolchain-identifier-A");
    assertThat(ccProviderA.getHostSystemName()).isEqualTo("host-system-name-A");
    assertThat(ccProviderA.getTargetGnuSystemName()).isEqualTo("target-system-name-A");
    assertThat(ccProviderA.getTargetCpu()).isEqualTo("k8");
    assertThat(ccProviderA.getTargetLibc()).isEqualTo("target-libc-A");
    assertThat(ccProviderA.getCompiler()).isEqualTo("compiler-A");
    assertThat(ccProviderA.getAbi()).isEqualTo("abi-version-A");
    assertThat(ccProviderA.getAbiGlibcVersion()).isEqualTo("abi-libc-version-A");
    assertThat(ccProviderA.getToolPathFragment(Tool.AR)).isEqualTo(getToolPath("path/to/ar-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.CPP)).isEqualTo(getToolPath("path/to/cpp-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.GCC)).isEqualTo(getToolPath("path/to/gcc-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.GCOV)).isEqualTo(getToolPath("path/to/gcov-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.LD)).isEqualTo(getToolPath("path/to/ld-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.NM)).isEqualTo(getToolPath("path/to/nm-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.OBJCOPY))
        .isEqualTo(getToolPath("path/to/objcopy-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.OBJDUMP))
        .isEqualTo(getToolPath("path/to/objdump-A"));
    assertThat(ccProviderA.getToolPathFragment(Tool.STRIP))
        .isEqualTo(getToolPath("path/to/strip-A"));
    assertThat(ccProviderA.supportsGoldLinker()).isTrue();
    assertThat(ccProviderA.supportsStartEndLib()).isTrue();
    assertThat(ccProviderA.supportsEmbeddedRuntimes()).isTrue();
    assertThat(ccProviderA.toolchainNeedsPic()).isTrue();

    assertThat(ccProviderA.getLegacyCompileOptionsWithCopts())
        .containsExactly(
            "compiler-flag-A-1", "compiler-flag-A-2", "fastbuild-flag-A-1", "fastbuild-flag-A-2")
        .inOrder();
    assertThat(ccProviderA.getUnfilteredCompilerOptions())
        .containsExactly("unfiltered-flag-A-1", "unfiltered-flag-A-2")
        .inOrder();
    assertThat(CppHelper.getDynamicLinkOptions(toolchainA, ccProviderA, true))
        .containsExactly(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "linker-fastbuild-flag-A-1",
            "linker-fastbuild-flag-A-2",
            "solinker-flag-A-1",
            "solinker-flag-A-2")
        .inOrder();

    // Only test a couple of compilation/linking mode combinations
    // (but test each mode at least once.)
    assertThat(
            ccProviderA.configureAllLegacyLinkOptions(
                CompilationMode.FASTBUILD, LinkingMode.LEGACY_FULLY_STATIC))
        .containsExactly(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "linker-fastbuild-flag-A-1",
            "linker-fastbuild-flag-A-2",
            "fully-static-flag-A-1",
            "fully-static-flag-A-2")
        .inOrder();
    assertThat(ccProviderA.configureAllLegacyLinkOptions(CompilationMode.DBG, LinkingMode.DYNAMIC))
        .containsExactly(
            "linker-flag-A-1", "linker-flag-A-2", "linker-dbg-flag-A-1", "linker-dbg-flag-A-2")
        .inOrder();
    assertThat(
            ccProviderA.configureAllLegacyLinkOptions(
                CompilationMode.OPT, LinkingMode.LEGACY_FULLY_STATIC))
        .containsExactly(
            "linker-flag-A-1", "linker-flag-A-2", "fully-static-flag-A-1", "fully-static-flag-A-2")
        .inOrder();

    assertThat(
            ccProviderA.configureAllLegacyLinkOptions(
                CompilationMode.OPT, LinkingMode.LEGACY_FULLY_STATIC))
        .containsExactly(
            "linker-flag-A-1", "linker-flag-A-2", "fully-static-flag-A-1", "fully-static-flag-A-2")
        .inOrder();

    assertThat(ccProviderA.getObjCopyOptionsForEmbedding())
        .containsExactly("objcopy-embed-flag-A-1", "objcopy-embed-flag-A-2")
        .inOrder();
    assertThat(ccProviderA.getLdOptionsForEmbedding())
        .containsExactly("ld-embed-flag-A-1", "ld-embed-flag-A-2")
        .inOrder();

    assertThat(makeProviderA.getVariables().entrySet())
        .containsAllIn(
            ImmutableMap.<String, String>builder()
                .put("SOME_MAKE_VARIABLE-A-1", "make-variable-value-A-1")
                .put("SOME_MAKE_VARIABLE-A-2", "make-variable-value-A-2 with spaces in")
                .put("STACK_FRAME_UNLIMITED", "")
                .build()
                .entrySet());
    assertThat(ccProviderA.getBuiltInIncludeDirectories())
        .containsExactly(
            getToolPath("system-include-dir-A-1"), getToolPath("system-include-dir-A-2"))
        .inOrder();

    // Cursory testing of the "B" toolchain only; assume that if none of
    // toolchain B bled through into toolchain A, the reverse also didn't occur. And
    // we test more of it with the "C" toolchain below.
    checkToolchainB(loader, "--cpu=k8", "--host_cpu=k8", "--compiler=compiler-B");
    checkToolchainB(
        loader, "--cpu=k8", "--host_cpu=k8", "--compiler=compiler-B", "--compilation_mode=opt");

    // Make sure nothing bled through to the nearly-empty "C" toolchain. This is also testing for
    // all the defaults.
    // Need to clear out the android cpu options to avoid this split transition in Bazel.
    CppConfiguration toolchainC =
        create(
            loader,
            "--compiler=compiler-C",
            "--cpu=k8",
            "--host_cpu=k8",
            "--android_cpu=",
            "--fat_apk_cpu=");
    CcToolchainProvider ccProviderC = getCcToolchainProvider(toolchainC);
    assertThat(ccProviderC.getToolchainIdentifier()).isEqualTo("toolchain-identifier-C");
    assertThat(ccProviderC.getHostSystemName()).isEqualTo("host-system-name-C");
    assertThat(ccProviderC.getTargetGnuSystemName()).isEqualTo("target-system-name-C");
    assertThat(ccProviderC.getTargetCpu()).isEqualTo("k8");
    assertThat(ccProviderC.getTargetLibc()).isEqualTo("target-libc-C");
    assertThat(ccProviderC.getCompiler()).isEqualTo("compiler-C");
    assertThat(ccProviderC.getAbi()).isEqualTo("abi-version-C");
    assertThat(ccProviderC.getAbiGlibcVersion()).isEqualTo("abi-libc-version-C");
    // Don't bother with testing the list of tools again.
    assertThat(ccProviderC.supportsGoldLinker()).isFalse();
    assertThat(ccProviderC.supportsStartEndLib()).isFalse();
    assertThat(ccProviderC.supportsInterfaceSharedObjects()).isFalse();
    assertThat(ccProviderC.supportsEmbeddedRuntimes()).isFalse();
    assertThat(ccProviderC.toolchainNeedsPic()).isFalse();
    assertThat(ccProviderC.supportsFission()).isFalse();

    assertThat(ccProviderC.getLegacyCompileOptionsWithCopts()).isEmpty();
    assertThat(ccProviderC.getUnfilteredCompilerOptions()).isEmpty();
    assertThat(CppHelper.getDynamicLinkOptions(toolchainC, ccProviderC, true)).isEmpty();
    assertThat(
            ccProviderC.configureAllLegacyLinkOptions(
                CompilationMode.FASTBUILD, LinkingMode.LEGACY_FULLY_STATIC))
        .isEmpty();
    assertThat(ccProviderC.configureAllLegacyLinkOptions(CompilationMode.DBG, LinkingMode.DYNAMIC))
        .isEmpty();
    assertThat(
            ccProviderC.configureAllLegacyLinkOptions(
                CompilationMode.OPT, LinkingMode.LEGACY_FULLY_STATIC))
        .isEmpty();
    assertThat(ccProviderC.getObjCopyOptionsForEmbedding()).isEmpty();
    assertThat(ccProviderC.getLdOptionsForEmbedding()).isEmpty();

    assertThat(ccProviderC.getAdditionalMakeVariables())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                "CC_FLAGS", "",
                "STACK_FRAME_UNLIMITED", ""));
    assertThat(ccProviderC.getBuiltInIncludeDirectories()).isEmpty();
    assertThat(ccProviderC.getSysroot()).isNull();
  }

  protected PathFragment getToolPath(String path) throws LabelSyntaxException {
    PackageIdentifier packageIdentifier =
        PackageIdentifier.create(
            TestConstants.TOOLS_REPOSITORY,
            PathFragment.create(TestConstants.MOCK_CC_CROSSTOOL_PATH).getRelative(path));
    return packageIdentifier.getPathUnderExecRoot();
  }

  private void checkToolchainB(CppConfigurationLoader loader, String... args) throws Exception {
    CppConfiguration toolchainB = create(loader, args);
    CcToolchainProvider ccProviderB = getCcToolchainProvider(toolchainB);
    assertThat(ccProviderB.getToolchainIdentifier()).isEqualTo("toolchain-identifier-B");
    assertThat(ccProviderB.configureAllLegacyLinkOptions(CompilationMode.DBG, LinkingMode.DYNAMIC))
        .containsExactly(
            "linker-flag-B-1", "linker-flag-B-2", "linker-dbg-flag-B-1", "linker-dbg-flag-B-2")
        .inOrder();
    assertThat(ccProviderB.getLegacyCompileOptionsWithCopts())
        .containsAllOf("compiler-flag-B-1", "compiler-flag-B-2")
        .inOrder();
  }

  /**
   * Tests that we can select a toolchain using the --compiler flag, as long as it selects a unique
   * result. Also tests the error messages we get when it doesn't.
   */
  @Test
  public void testToolchainSelection() throws Exception {
    CppConfigurationLoader loader =
        loader(
            // Needs to include \n's; as a single line it hits a parser limitation.
            "major_version: \"12\"\n"
                + "minor_version: \"0\"\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-A\"\n"
                + "  host_system_name: \"host-system-name-A\"\n"
                + "  target_system_name: \"target-system-name-A\"\n"
                + "  target_cpu: \"banana_cpu\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-C\"\n"
                + "  host_system_name: \"host-system-name-C\"\n"
                + "  target_system_name: \"target-system-name-C\"\n"
                + "  target_cpu: \"banana_cpu\"\n"
                + "  target_libc: \"target-libc-C\"\n"
                + "  compiler: \"compiler-C\"\n"
                + "  abi_version: \"abi-version-C\"\n"
                + "  abi_libc_version: \"abi-libc-version-C\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-A-avocado_cpu\"\n"
                + "  host_system_name: \"host-system-name-A\"\n"
                + "  target_system_name: \"target-system-name-A\"\n"
                + "  target_cpu: \"avocado_cpu\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-B-avocado_cpu\"\n"
                + "  host_system_name: \"host-system-name-A\"\n"
                + "  target_system_name: \"target-system-name-A\"\n"
                + "  target_cpu: \"avocado_cpu\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-B\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-B\"\n"
                + "  host_system_name: \"host-system-name-B\"\n"
                + "  target_system_name: \"target-system-name-B\"\n"
                + "  target_cpu: \"banana_cpu\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-B\"\n"
                + "  abi_version: \"abi-version-B\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}");

    // Uses banana_cpu alone in cc_toolchain_suite.toolchains.
    assertThat(create(loader, "--cpu=banana_cpu").getToolchainIdentifier())
        .isEqualTo("toolchain-identifier-A");
    // Uses avocado_cpu alone in cc_toolchain_suite.toolchains
    assertThat(create(loader, "--cpu=avocado_cpu").getToolchainIdentifier())
        .isEqualTo("toolchain-identifier-A-avocado_cpu");

    // We can select the unique avocado_cpu toolchain with its compiler.
    assertThat(
            create(loader, "--cpu=avocado_cpu", "--compiler=compiler-B").getToolchainIdentifier())
        .isEqualTo("toolchain-identifier-B-avocado_cpu");

    try {
      create(loader, "--cpu=banana_cpu", "--compiler=nonexistent-compiler");
      fail("Expected an error that no toolchain matched.");
    } catch (InvalidConfigurationException e) {
      String message =
          "does not contain a toolchain for cpu 'banana_cpu' and compiler 'nonexistent-compiler'";
      assertThat(e).hasMessageThat().contains(message);
    }
  }

  private void assertStringStartsWith(String expected, String text) {
    if (!text.startsWith(expected)) {
      fail("expected <" + expected + ">, but got <" + text + ">");
    }
  }

  @Test
  public void testIncompleteFile() throws Exception {
    try {
      CrosstoolConfigurationLoader.getUncachedReleaseConfiguration(
          "/CROSSTOOL", () -> "major_version: \"12\"");
      fail();
    } catch (InvalidConfigurationException e) {
      assertStringStartsWith(
          "Could not read the crosstool configuration file "
              + "'/CROSSTOOL', because of an incomplete protocol buffer",
          e.getMessage());
    }
  }

  /**
   * Returns a test crosstool config with the specified tool missing from the tool_path
   * set. Also allows injection of custom fields.
   */
  private static String getConfigWithMissingToolDef(Tool missingTool, String... customFields) {
    StringBuilder s =
        new StringBuilder(
            "major_version: \"12\""
                + "minor_version: \"0\""
                + "toolchain {"
                + "  toolchain_identifier: \"toolchain-identifier\""
                + "  host_system_name: \"host-system-name\""
                + "  target_system_name: \"target-system-name\""
                + "  target_cpu: \"banana_cpu\""
                + "  target_libc: \"target-libc\""
                + "  compiler: \"compiler\""
                + "  abi_version: \"abi-version\""
                + "  abi_libc_version: \"abi-libc-version\"");

    for (String customField : customFields) {
      s.append(customField);
    }
    for (Tool tool : Tool.values()) {
      if (tool != missingTool) {
        String toolName = tool.getNamePart();
        s.append("  tool_path { name: \"" + toolName + "\" path: \"path-to-" + toolName + "\" }");
      }
    }
    s.append("}");
    return s.toString();
  }

  /**
   * For a non-fission-supporting crosstool, there's no need to check the dwp tool path.
   */
  @Test
  public void testNonFissionConfigWithMissingDwp() throws Exception {
    CppConfigurationLoader loader =
        loader(getConfigWithMissingToolDef(Tool.DWP, "supports_fission: false"));
    // The following line throws an IllegalArgumentException if an expected tool path is missing.
    create(loader, "--cpu=banana_cpu");
  }

  @Test
  public void testInvalidFile() throws Exception {
    try {
      CrosstoolConfigurationLoader.getUncachedReleaseConfiguration(
          "/CROSSTOOL", () -> "some xxx : yak \"");
      fail();
    } catch (InvalidConfigurationException e) {
      assertStringStartsWith(
          "Could not read the crosstool configuration file "
              + "'/CROSSTOOL', because of a parser error",
          e.getMessage());
    }
  }

  /**
   * Tests interpretation of static_runtimes_filegroup / dynamic_runtimes_filegroup.
   */
  @Test
  public void testCustomRuntimeLibraryPaths() throws Exception {
    CppConfigurationLoader loader =
        loader(
            "major_version: \"v17\""
                + "minor_version: \"0\""
                + "toolchain {" // "default-libs": runtime libraries in default locations.
                + "  toolchain_identifier: \"default-libs\""
                + "  host_system_name: \"host-system-name\""
                + "  target_system_name: \"target-system-name\""
                + "  target_cpu: \"k8\""
                + "  target_libc: \"target-libc\""
                + "  compiler: \"compiler\""
                + "  abi_version: \"abi-version\""
                + "  abi_libc_version: \"abi-libc-version\""
                + "  supports_embedded_runtimes: true"
                + "}\n"
                + "toolchain {" // "custom-libs" runtime libraries in toolchain-specified locations.
                + "  toolchain_identifier: \"custom-libs\""
                + "  host_system_name: \"host-system-name\""
                + "  target_system_name: \"target-system-name\""
                + "  target_cpu: \"piii\""
                + "  target_libc: \"target-libc\""
                + "  compiler: \"compiler\""
                + "  abi_version: \"abi-version\""
                + "  abi_libc_version: \"abi-libc-version\""
                + "  supports_embedded_runtimes: true"
                + "  static_runtimes_filegroup: \"static-group\""
                + "  dynamic_runtimes_filegroup: \"dynamic-group\""
                + "}\n");

    PackageIdentifier ctTop = MockCcSupport.getMockCrosstoolsTop();
    if (ctTop.getRepository().isDefault()) {
      ctTop = PackageIdentifier.createInMainRepo(ctTop.getPackageFragment());
    }
    CppConfiguration defaultLibs = create(loader, "--cpu=k8", "--host_cpu=k8");
    CcToolchainProvider defaultLibsToolchain = getCcToolchainProvider(defaultLibs);
    assertThat(defaultLibsToolchain.getStaticRuntimeLibsLabel())
        .isEqualTo(Label.create(ctTop, "static-runtime-libs-k8"));
    assertThat(defaultLibsToolchain.getDynamicRuntimeLibsLabel())
        .isEqualTo(Label.create(ctTop, "dynamic-runtime-libs-k8"));

    CppConfiguration customLibs = create(loader, "--cpu=piii", "--host_cpu=k8");
    CcToolchainProvider customLibsToolchain = getCcToolchainProvider(customLibs);
    assertThat(customLibsToolchain.getStaticRuntimeLibsLabel())
        .isEqualTo(Label.create(ctTop, "static-group"));
    assertThat(customLibsToolchain.getDynamicRuntimeLibsLabel())
        .isEqualTo(Label.create(ctTop, "dynamic-group"));
  }
}
