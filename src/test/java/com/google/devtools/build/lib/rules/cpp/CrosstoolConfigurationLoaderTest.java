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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link CppConfigurationLoader}.
 */
@RunWith(JUnit4.class)
public class CrosstoolConfigurationLoaderTest extends AnalysisTestCase {
  private static final Collection<String> NO_FEATURES = Collections.emptySet();

  private BuildOptions createBuildOptionsForTest(String... args) {
    ImmutableList<Class<? extends FragmentOptions>> testFragments =
        TestRuleClassProvider.getRuleClassProvider().getConfigurationOptions();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(testFragments);
    try {
      optionsParser.parse(args);
      InvocationPolicyEnforcer optionsPolicyEnforcer = analysisMock.getInvocationPolicyEnforcer();
      optionsPolicyEnforcer.enforce(optionsParser);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
    return BuildOptions.applyStaticConfigOverride(BuildOptions.of(testFragments, optionsParser));
  }

  private CppConfiguration create(CppConfigurationLoader loader, String... args) throws Exception {
    ConfigurationEnvironment env =
        new ConfigurationEnvironment.TargetProviderEnvironment(
            skyframeExecutor.getPackageManager(), reporter, directories);
    return loader.create(env, createBuildOptionsForTest(args));
  }

  private CppConfigurationLoader loader(String crosstoolFileContents) throws IOException {
    getAnalysisMock().ccSupport().setupCrosstoolWithRelease(mockToolsConfig, crosstoolFileContents);
    return new CppConfigurationLoader(Functions.<String>identity());
  }

  private CppConfigurationLoader loaderWithOptionalTool(String optionalTool) throws IOException {
    return loader(
        "major_version: \"12\""
            + "minor_version: \"0\""
            + "default_target_cpu: \"cpu\""
            + "default_toolchain {"
            + "  cpu: \"cpu\""
            + "  toolchain_identifier: \"toolchain-identifier\""
            + "}"
            + "toolchain {"
            + "  toolchain_identifier: \"toolchain-identifier\""
            + "  host_system_name: \"host-system-name\""
            + "  target_system_name: \"target-system-name\""
            + "  target_cpu: \"piii\""
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

  /**
   * Checks that we do not accidentally change the proto format in incompatible
   * ways. Do not modify the configuration file in this test, except if you are
   * absolutely certain that it is backwards-compatible.
   */
  @Test
  public void testSimpleCompleteConfiguration() throws Exception {
    CppConfigurationLoader loader = loaderWithOptionalTool("");

    CppConfiguration toolchain = create(loader, "--cpu=cpu");
    assertEquals("toolchain-identifier", toolchain.getToolchainIdentifier());

    assertEquals("host-system-name", toolchain.getHostSystemName());
    assertEquals("compiler", toolchain.getCompiler());
    assertEquals("target-libc", toolchain.getTargetLibc());
    assertEquals("piii", toolchain.getTargetCpu());
    assertEquals("target-system-name", toolchain.getTargetGnuSystemName());

    assertEquals(getToolPath("/path-to-ar"), toolchain.getToolPathFragment(Tool.AR));

    assertEquals("abi-version", toolchain.getAbi());
    assertEquals("abi-libc-version", toolchain.getAbiGlibcVersion());

    assertTrue(toolchain.supportsGoldLinker());
    assertFalse(toolchain.supportsStartEndLib());
    assertFalse(toolchain.supportsInterfaceSharedObjects());
    assertFalse(toolchain.supportsEmbeddedRuntimes());
    assertFalse(toolchain.toolchainNeedsPic());
    assertTrue(toolchain.supportsFission());

    assertEquals(
        ImmutableList.of(getToolPath("/system-include-dir")),
        toolchain.getBuiltInIncludeDirectories());
    assertNull(toolchain.getSysroot());

    assertEquals(Arrays.asList("c", "fastbuild"), toolchain.getCompilerOptions(NO_FEATURES));
    assertEquals(Arrays.<String>asList(), toolchain.getCOptions());
    assertEquals(Arrays.asList("cxx", "cxx-fastbuild"), toolchain.getCxxOptions(NO_FEATURES));
    assertEquals(Arrays.asList("unfiltered"), toolchain.getUnfilteredCompilerOptions(NO_FEATURES));

    assertEquals(Arrays.<String>asList(), toolchain.getLinkOptions());
    assertEquals(
        Arrays.asList("linker", "linker-fastbuild", "fully static"),
        toolchain.getFullyStaticLinkOptions(NO_FEATURES, false));
    assertEquals(
        Arrays.asList("linker", "linker-fastbuild", "dynamic"),
        toolchain.getDynamicLinkOptions(NO_FEATURES, false));
    assertEquals(
        Arrays.asList("linker", "linker-fastbuild", "mostly static", "solinker"),
        toolchain.getFullyStaticLinkOptions(NO_FEATURES, true));
    assertEquals(
        Arrays.asList("linker", "linker-fastbuild", "dynamic", "solinker"),
        toolchain.getDynamicLinkOptions(NO_FEATURES, true));

    assertEquals(Arrays.asList("objcopy"), toolchain.getObjCopyOptionsForEmbedding());
    assertEquals(Arrays.<String>asList(), toolchain.getLdOptionsForEmbedding());
    assertEquals(Arrays.asList("rcsD"), toolchain.getArFlags());

    assertThat(toolchain.getAdditionalMakeVariables().entrySet())
        .containsExactlyElementsIn(
            ImmutableMap.of(
                    "SOME_MAKE_VARIABLE", "make-variable-value",
                    "STACK_FRAME_UNLIMITED", "",
                    "CC_FLAGS", "")
                .entrySet());

    assertEquals(getToolPath("/path-to-ld"), toolchain.getLdExecutable());
    assertEquals(getToolPath("/path-to-dwp"), toolchain.getToolPathFragment(Tool.DWP));
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
                + "default_target_cpu: \"piii\"\n"
                + "default_toolchain {\n"
                + "  cpu: \"piii\"\n"
                + "  toolchain_identifier: \"toolchain-identifier-A\"\n"
                + "}\n"
                + "default_toolchain {\n"
                + "  cpu: \"k8\"\n"
                + "  toolchain_identifier: \"toolchain-identifier-B\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-A\"\n"
                + "  host_system_name: \"host-system-name-A\"\n"
                + "  target_system_name: \"target-system-name-A\"\n"
                + "  target_cpu: \"piii\"\n"
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
                + "  ar_flag : \"ar-flag-A\"\n"
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
                + "  default_grte_top: \"//some\""
                + "  debian_extra_requires: \"a\""
                + "  debian_extra_requires: \"b\""
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-B\"\n"
                + "  host_system_name: \"host-system-name-B\"\n"
                + "  target_system_name: \"target-system-name-B\"\n"
                + "  target_cpu: \"piii\"\n"
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
                + "  optional_compiler_flag {\n"
                + "    default_setting_name: \"crosstool_fig\"\n"
                + "    flag: \"-Wfig\"\n"
                + "  }\n"
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
                + "  ar_flag : \"ar-flag-B\"\n"
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
                + "  lipo_mode_flags {"
                + "    mode: OFF"
                + "    compiler_flag: \"lipo_off\""
                + "    cxx_flag: \"cxx-lipo_off\""
                + "    linker_flag: \"linker-lipo_off\""
                + "  }"
                + "  lipo_mode_flags {"
                + "    mode: BINARY"
                + "    compiler_flag: \"lipo_binary\""
                + "    cxx_flag: \"cxx-lipo_binary\""
                + "    linker_flag: \"linker-lipo_binary\""
                + "  }"
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
                + "  default_grte_top: \"//some\"\n"
                + "  debian_extra_requires: \"c\""
                + "  debian_extra_requires: \"d\""
                + "}\n"
                + "default_setting {\n"
                + "  name: \"crosstool_fig\"\n"
                + "  default_value: false\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-C\"\n"
                + "  host_system_name: \"host-system-name-C\"\n"
                + "  target_system_name: \"target-system-name-C\"\n"
                + "  target_cpu: \"piii\"\n"
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

    CppConfiguration toolchainA = create(loader, "--cpu=piii");
    assertEquals("toolchain-identifier-A", toolchainA.getToolchainIdentifier());
    assertEquals("host-system-name-A", toolchainA.getHostSystemName());
    assertEquals("target-system-name-A", toolchainA.getTargetGnuSystemName());
    assertEquals("piii", toolchainA.getTargetCpu());
    assertEquals("target-libc-A", toolchainA.getTargetLibc());
    assertEquals("compiler-A", toolchainA.getCompiler());
    assertEquals("abi-version-A", toolchainA.getAbi());
    assertEquals("abi-libc-version-A", toolchainA.getAbiGlibcVersion());
    assertEquals(getToolPath("path/to/ar-A"), toolchainA.getToolPathFragment(Tool.AR));
    assertEquals(getToolPath("path/to/cpp-A"), toolchainA.getToolPathFragment(Tool.CPP));
    assertEquals(getToolPath("path/to/gcc-A"), toolchainA.getToolPathFragment(Tool.GCC));
    assertEquals(getToolPath("path/to/gcov-A"), toolchainA.getToolPathFragment(Tool.GCOV));
    assertEquals(getToolPath("path/to/ld-A"), toolchainA.getToolPathFragment(Tool.LD));
    assertEquals(getToolPath("path/to/nm-A"), toolchainA.getToolPathFragment(Tool.NM));
    assertEquals(getToolPath("path/to/objcopy-A"), toolchainA.getToolPathFragment(Tool.OBJCOPY));
    assertEquals(getToolPath("path/to/objdump-A"), toolchainA.getToolPathFragment(Tool.OBJDUMP));
    assertEquals(getToolPath("path/to/strip-A"), toolchainA.getToolPathFragment(Tool.STRIP));
    assertTrue(toolchainA.supportsGoldLinker());
    assertTrue(toolchainA.supportsStartEndLib());
    assertTrue(toolchainA.supportsEmbeddedRuntimes());
    assertTrue(toolchainA.toolchainNeedsPic());

    assertEquals(
        Arrays.asList(
            "compiler-flag-A-1", "compiler-flag-A-2", "fastbuild-flag-A-1", "fastbuild-flag-A-2"),
        toolchainA.getCompilerOptions(NO_FEATURES));
    assertEquals(
        Arrays.asList(
            "cxx-flag-A-1", "cxx-flag-A-2", "cxx-fastbuild-flag-A-1", "cxx-fastbuild-flag-A-2"),
        toolchainA.getCxxOptions(NO_FEATURES));
    assertEquals(
        Arrays.asList("unfiltered-flag-A-1", "unfiltered-flag-A-2"),
        toolchainA.getUnfilteredCompilerOptions(NO_FEATURES));
    assertEquals(
        Arrays.asList(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "linker-fastbuild-flag-A-1",
            "linker-fastbuild-flag-A-2",
            "solinker-flag-A-1",
            "solinker-flag-A-2"),
        toolchainA.getDynamicLinkOptions(NO_FEATURES, true));

    // Only test a couple of compilation/lipo/linking mode combinations
    // (but test each mode at least once.)
    assertEquals(
        Arrays.asList(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "linker-fastbuild-flag-A-1",
            "linker-fastbuild-flag-A-2",
            "fully-static-flag-A-1",
            "fully-static-flag-A-2"),
        toolchainA.configureLinkerOptions(
            CompilationMode.FASTBUILD,
            LipoMode.OFF,
            LinkingMode.FULLY_STATIC,
            new PathFragment("hello-world/ld")));
    assertEquals(
        Arrays.asList(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "linker-dbg-flag-A-1",
            "linker-dbg-flag-A-2"),
        toolchainA.configureLinkerOptions(
            CompilationMode.DBG,
            LipoMode.OFF,
            LinkingMode.DYNAMIC,
            new PathFragment("hello-world/ld")));
    assertEquals(
        Arrays.asList(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "fully-static-flag-A-1",
            "fully-static-flag-A-2"),
        toolchainA.configureLinkerOptions(
            CompilationMode.OPT,
            LipoMode.OFF,
            LinkingMode.FULLY_STATIC,
            new PathFragment("hello-world/ld")));

    assertEquals(
        Arrays.asList(
            "linker-flag-A-1",
            "linker-flag-A-2",
            "fully-static-flag-A-1",
            "fully-static-flag-A-2"),
        toolchainA.configureLinkerOptions(
            CompilationMode.OPT,
            LipoMode.BINARY,
            LinkingMode.FULLY_STATIC,
            new PathFragment("hello-world/ld")));

    assertEquals(
        Arrays.asList("objcopy-embed-flag-A-1", "objcopy-embed-flag-A-2"),
        toolchainA.getObjCopyOptionsForEmbedding());
    assertEquals(
        Arrays.asList("ld-embed-flag-A-1", "ld-embed-flag-A-2"),
        toolchainA.getLdOptionsForEmbedding());
    assertEquals(Arrays.asList("ar-flag-A"), toolchainA.getArFlags());

    assertThat(toolchainA.getAdditionalMakeVariables().entrySet())
        .containsExactlyElementsIn(
            ImmutableMap.<String, String>builder()
                .put("SOME_MAKE_VARIABLE-A-1", "make-variable-value-A-1")
                .put("SOME_MAKE_VARIABLE-A-2", "make-variable-value-A-2 with spaces in")
                .put("CC_FLAGS", "--sysroot=some")
                .put("STACK_FRAME_UNLIMITED", "")
                .build()
                .entrySet());
    assertEquals(
        Arrays.asList(
            getToolPath("/system-include-dir-A-1"), getToolPath("/system-include-dir-A-2")),
        toolchainA.getBuiltInIncludeDirectories());
    assertEquals(new PathFragment("some"), toolchainA.getSysroot());

    // Cursory testing of the "B" toolchain only; assume that if none of
    // toolchain B bled through into toolchain A, the reverse also didn't occur. And
    // we test more of it with the "C" toolchain below.
    checkToolchainB(loader, LipoMode.OFF, "--cpu=k8", "--lipo=off");
    checkToolchainB(loader, LipoMode.BINARY, "--cpu=k8", "--lipo=binary");

    // Make sure nothing bled through to the nearly-empty "C" toolchain. This is also testing for
    // all the defaults.
    CppConfiguration toolchainC =
        create(loader, "--compiler=compiler-C", "--glibc=target-libc-C", "--cpu=piii");
    assertEquals("toolchain-identifier-C", toolchainC.getToolchainIdentifier());
    assertEquals("host-system-name-C", toolchainC.getHostSystemName());
    assertEquals("target-system-name-C", toolchainC.getTargetGnuSystemName());
    assertEquals("piii", toolchainC.getTargetCpu());
    assertEquals("target-libc-C", toolchainC.getTargetLibc());
    assertEquals("compiler-C", toolchainC.getCompiler());
    assertEquals("abi-version-C", toolchainC.getAbi());
    assertEquals("abi-libc-version-C", toolchainC.getAbiGlibcVersion());
    // Don't bother with testing the list of tools again.
    assertFalse(toolchainC.supportsGoldLinker());
    assertFalse(toolchainC.supportsStartEndLib());
    assertFalse(toolchainC.supportsInterfaceSharedObjects());
    assertFalse(toolchainC.supportsEmbeddedRuntimes());
    assertFalse(toolchainC.toolchainNeedsPic());
    assertFalse(toolchainC.supportsFission());

    assertThat(toolchainC.getCompilerOptions(NO_FEATURES)).isEmpty();
    assertThat(toolchainC.getCOptions()).isEmpty();
    assertThat(toolchainC.getCxxOptions(NO_FEATURES)).isEmpty();
    assertThat(toolchainC.getUnfilteredCompilerOptions(NO_FEATURES)).isEmpty();
    assertEquals(Collections.EMPTY_LIST, toolchainC.getDynamicLinkOptions(NO_FEATURES, true));
    assertEquals(
        Collections.EMPTY_LIST,
        toolchainC.configureLinkerOptions(
            CompilationMode.FASTBUILD,
            LipoMode.OFF,
            LinkingMode.FULLY_STATIC,
            new PathFragment("hello-world/ld")));
    assertEquals(
        Collections.EMPTY_LIST,
        toolchainC.configureLinkerOptions(
            CompilationMode.DBG,
            LipoMode.OFF,
            LinkingMode.DYNAMIC,
            new PathFragment("hello-world/ld")));
    assertEquals(
        Collections.EMPTY_LIST,
        toolchainC.configureLinkerOptions(
            CompilationMode.OPT,
            LipoMode.OFF,
            LinkingMode.FULLY_STATIC,
            new PathFragment("hello-world/ld")));
    assertThat(toolchainC.getObjCopyOptionsForEmbedding()).isEmpty();
    assertThat(toolchainC.getLdOptionsForEmbedding()).isEmpty();

    assertThat(toolchainC.getAdditionalMakeVariables().entrySet())
        .containsExactlyElementsIn(
            ImmutableMap.<String, String>builder()
                .put("CC_FLAGS", "")
                .put("STACK_FRAME_UNLIMITED", "")
                .build()
                .entrySet());
    assertThat(toolchainC.getBuiltInIncludeDirectories()).isEmpty();
    assertNull(toolchainC.getSysroot());
  }

  protected PathFragment getToolPath(String path) throws LabelSyntaxException {
    PackageIdentifier packageIdentifier =
        PackageIdentifier.create(
            TestConstants.TOOLS_REPOSITORY,
            new PathFragment(
                new PathFragment(TestConstants.TOOLS_REPOSITORY_PATH), new PathFragment(path)));
    return packageIdentifier.getPathUnderExecRoot();
  }

  private void checkToolchainB(CppConfigurationLoader loader, LipoMode lipoMode, String... args)
      throws Exception {
    String lipoSuffix = lipoMode.toString().toLowerCase();
    CppConfiguration toolchainB = create(loader, args);
    assertEquals("toolchain-identifier-B", toolchainB.getToolchainIdentifier());
    assertEquals(
        Arrays.asList(
            "linker-flag-B-1",
            "linker-flag-B-2",
            "linker-dbg-flag-B-1",
            "linker-dbg-flag-B-2",
            "linker-lipo_" + lipoSuffix),
        toolchainB.configureLinkerOptions(
            CompilationMode.DBG,
            lipoMode,
            LinkingMode.DYNAMIC,
            new PathFragment("hello-world/ld")));
    assertEquals(
        ImmutableList.<String>of(
            "compiler-flag-B-1",
            "compiler-flag-B-2",
            "fastbuild-flag-B-1",
            "fastbuild-flag-B-2",
            "lipo_" + lipoSuffix,
            "-Wfig"),
        toolchainB.getCompilerOptions(ImmutableList.of("crosstool_fig")));
  }

  /**
   * Tests that we can select a toolchain using a subset of the --compiler and
   * --glibc flags, as long as they select a unique result. Also tests the error
   * messages we get when they don't.
   */
  @Test
  public void testCompilerLibcSearch() throws Exception {
    CppConfigurationLoader loader =
        loader(
            // Needs to include \n's; as a single line it hits a parser limitation.
            "major_version: \"12\"\n"
                + "minor_version: \"0\"\n"
                + "default_target_cpu: \"k8\"\n"
                + "default_toolchain {\n"
                + "  cpu: \"piii\"\n"
                + "  toolchain_identifier: \"toolchain-identifier-AA-piii\"\n"
                + "}\n"
                + "default_toolchain {\n"
                + "  cpu: \"k8\"\n"
                + "  toolchain_identifier: \"toolchain-identifier-BB\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-AA\"\n"
                + "  host_system_name: \"host-system-name-AA\"\n"
                + "  target_system_name: \"target-system-name-AA\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                // AA-piii is uniquely determined by libc and compiler.
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-AA-piii\"\n"
                + "  host_system_name: \"host-system-name-AA\"\n"
                + "  target_system_name: \"target-system-name-AA\"\n"
                + "  target_cpu: \"piii\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-AB\"\n"
                + "  host_system_name: \"host-system-name-AB\"\n"
                + "  target_system_name: \"target-system-name-AB\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-A\"\n"
                + "  compiler: \"compiler-B\"\n"
                + "  abi_version: \"abi-version-B\"\n"
                + "  abi_libc_version: \"abi-libc-version-A\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-BA\"\n"
                + "  host_system_name: \"host-system-name-BA\"\n"
                + "  target_system_name: \"target-system-name-BA\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-B\"\n"
                + "  compiler: \"compiler-A\"\n"
                + "  abi_version: \"abi-version-A\"\n"
                + "  abi_libc_version: \"abi-libc-version-B\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-BB\"\n"
                + "  host_system_name: \"host-system-name-BB\"\n"
                + "  target_system_name: \"target-system-name-BB\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-B\"\n"
                + "  compiler: \"compiler-B\"\n"
                + "  abi_version: \"abi-version-B\"\n"
                + "  abi_libc_version: \"abi-libc-version-B\"\n"
                + "}\n"
                + "toolchain {\n"
                + "  toolchain_identifier: \"toolchain-identifier-BC\"\n"
                + "  host_system_name: \"host-system-name-BC\"\n"
                + "  target_system_name: \"target-system-name-BC\"\n"
                + "  target_cpu: \"k8\"\n"
                + "  target_libc: \"target-libc-B\"\n"
                + "  compiler: \"compiler-C\"\n"
                + "  abi_version: \"abi-version-C\"\n"
                + "  abi_libc_version: \"abi-libc-version-B\"\n"
                + "}");

    // Uses the default toolchain for k8.
    assertEquals("toolchain-identifier-BB", create(loader, "--cpu=k8").getToolchainIdentifier());
    // Does not default to --cpu=k8; if no --cpu flag is present, Bazel defaults to the host cpu!
    assertEquals(
        "toolchain-identifier-BA",
        create(loader, "--cpu=k8", "--compiler=compiler-A", "--glibc=target-libc-B")
            .getToolchainIdentifier());
    // Uses the default toolchain for piii.
    assertEquals(
        "toolchain-identifier-AA-piii", create(loader, "--cpu=piii").getToolchainIdentifier());

    // We can select the unique piii toolchain with either its compiler or glibc.
    assertEquals(
        "toolchain-identifier-AA-piii",
        create(loader, "--cpu=piii", "--compiler=compiler-A").getToolchainIdentifier());
    assertEquals(
        "toolchain-identifier-AA-piii",
        create(loader, "--cpu=piii", "--glibc=target-libc-A").getToolchainIdentifier());

    // compiler-C uniquely identifies a toolchain, so we can use it.
    assertEquals(
        "toolchain-identifier-BC",
        create(loader, "--cpu=k8", "--compiler=compiler-C").getToolchainIdentifier());

    try {
      create(loader, "--cpu=k8", "--compiler=nonexistent-compiler");
      fail("Expected an error that no toolchain matched.");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessage(
              "No toolchain found for --cpu='k8' --compiler='nonexistent-compiler'. "
                  + "Valid toolchains are: [\n"
                  + "  --cpu='k8' --compiler='compiler-A' --glibc='target-libc-A',\n"
                  + "  --cpu='piii' --compiler='compiler-A' --glibc='target-libc-A',\n"
                  + "  --cpu='k8' --compiler='compiler-B' --glibc='target-libc-A',\n"
                  + "  --cpu='k8' --compiler='compiler-A' --glibc='target-libc-B',\n"
                  + "  --cpu='k8' --compiler='compiler-B' --glibc='target-libc-B',\n"
                  + "  --cpu='k8' --compiler='compiler-C' --glibc='target-libc-B',\n"
                  + "]");
    }

    try {
      create(loader, "--cpu=k8", "--glibc=target-libc-A");
      fail("Expected an error that multiple toolchains matched.");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessage(
              "Multiple toolchains found for --cpu='k8' --glibc='target-libc-A': [\n"
                  + "  --cpu='k8' --compiler='compiler-A' --glibc='target-libc-A',\n"
                  + "  --cpu='k8' --compiler='compiler-B' --glibc='target-libc-A',\n"
                  + "]");
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
      CrosstoolConfigurationLoader.toReleaseConfiguration("/CROSSTOOL", "major_version: \"12\"");
      fail();
    } catch (IOException e) {
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
                + "default_target_cpu: \"cpu\""
                + "default_toolchain {"
                + "  cpu: \"cpu\""
                + "  toolchain_identifier: \"toolchain-identifier\""
                + "}"
                + "toolchain {"
                + "  toolchain_identifier: \"toolchain-identifier\""
                + "  host_system_name: \"host-system-name\""
                + "  target_system_name: \"target-system-name\""
                + "  target_cpu: \"piii\""
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

  @Test
  public void testConfigWithMissingToolDefs() throws Exception {
    CppConfigurationLoader loader = loader(getConfigWithMissingToolDef(Tool.STRIP));
    try {
      create(loader, "--cpu=cpu");
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("Tool path for 'strip' is missing");
    }
  }

  /**
   * For a fission-supporting crosstool: check the dwp tool path.
   */
  @Test
  public void testFissionConfigWithMissingDwp() throws Exception {
    CppConfigurationLoader loader =
        loader(getConfigWithMissingToolDef(Tool.DWP, "supports_fission: true"));
    try {
      create(loader, "--cpu=cpu");
      fail("Expected failed check on 'dwp' tool path");
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("Tool path for 'dwp' is missing");
    }
  }

  /**
   * For a non-fission-supporting crosstool, there's no need to check the dwp tool path.
   */
  @Test
  public void testNonFissionConfigWithMissingDwp() throws Exception {
    CppConfigurationLoader loader =
        loader(getConfigWithMissingToolDef(Tool.DWP, "supports_fission: false"));
    // The following line throws an IllegalArgumentException if an expected tool path is missing.
    create(loader, "--cpu=cpu");
  }

  @Test
  public void testInvalidFile() throws Exception {
    try {
      CrosstoolConfigurationLoader.toReleaseConfiguration("/CROSSTOOL", "some xxx : yak \"");
      fail();
    } catch (IOException e) {
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
                + "default_target_cpu: \"cpu\""
                + "default_toolchain {"
                + "  cpu: \"piii\""
                + "  toolchain_identifier: \"default-libs\""
                + "}"
                + "default_toolchain {"
                + "  cpu: \"k8\""
                + "  toolchain_identifier: \"custom-libs\""
                + "}"
                + "toolchain {" // "default-libs": runtime libraries in default locations.
                + "  toolchain_identifier: \"default-libs\""
                + "  host_system_name: \"host-system-name\""
                + "  target_system_name: \"target-system-name\""
                + "  target_cpu: \"piii\""
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
                + "  target_cpu: \"k8\""
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
    CppConfiguration defaultLibs = create(loader, "--cpu=piii");
    assertEquals(
        Label.create(ctTop, "static-runtime-libs-piii"), defaultLibs.getStaticRuntimeLibsLabel());
    assertEquals(
        Label.create(ctTop, "dynamic-runtime-libs-piii"), defaultLibs.getDynamicRuntimeLibsLabel());

    CppConfiguration customLibs = create(loader, "--cpu=k8");
    assertEquals(Label.create(ctTop, "static-group"), customLibs.getStaticRuntimeLibsLabel());
    assertEquals(Label.create(ctTop, "dynamic-group"), customLibs.getDynamicRuntimeLibsLabel());
  }

  /*
   * Crosstools should load fine with or without 'gcov-tool'. Those that define 'gcov-tool'
   * should also add a make variable.
   */
  @Test
  public void testOptionalGcovTool() throws Exception {
    // Crosstool with gcov-tool
    CppConfigurationLoader loader =
        loaderWithOptionalTool("  tool_path { name: \"gcov-tool\" path: \"path-to-gcov-tool\" }");
    CppConfiguration cppConfig = create(loader, "--cpu=cpu");
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    cppConfig.addGlobalMakeVariables(builder);
    assertNotNull(builder.build().get("GCOVTOOL"));

    // Crosstool without gcov-tool
    loader = loaderWithOptionalTool("");
    cppConfig = create(loader, "--cpu=cpu");
    builder = ImmutableMap.builder();
    cppConfig.addGlobalMakeVariables(builder);
    assertThat(builder.build()).doesNotContainKey("GCOVTOOL");
  }
}
