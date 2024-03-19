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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for shared library linking {@link CppLinkAction}.
 */
@RunWith(JUnit4.class)
public final class LibraryLinkingTest extends BuildViewTestCase {
  private List<String> getLinkOpts(CppLinkAction linkAction, String... optionPatterns)
      throws Exception {
    // Strip the first parameters from the argv, which are the dynamic library script
    // (usually tools/cpp/link_dynamic_library.sh), and its arguments.
    return linkAction.getArguments().subList(1, optionPatterns.length + 1);
  }

  private void assertLinkopts(CppLinkAction linkAction, String... optionPatterns) throws Exception {
    List<String> linkopts = getLinkOpts(linkAction, optionPatterns);
    for (int i = 0; i < optionPatterns.length; i++) {
      assertThat(linkopts.get(i)).matches(optionPatterns[i]);
    }
  }

  @Test
  public void testGeneratedLib() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));

    useConfiguration(
        "--platforms=" + TestConstants.PLATFORM_LABEL,
        "--noincompatible_remove_legacy_whole_archive");
    ConfiguredTarget genlib =
        scratchConfiguredTarget(
            "genrule",
            "thebinary.so",
            "genrule(name = 'genlib',",
            "        outs = ['genlib.a'],",
            "        cmd = '')",
            "cc_library(name = 'thelib',",
            "           srcs = [':genlib'],",
            "           linkstatic = 1)",
            "cc_binary(name = 'thebinary.so',",
            "          deps = [':thelib'],",
            "          linkstatic = 1,",
            "          linkshared = 1)");
    Artifact executable = getExecutable(genlib);
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(executable);
    assertLinkopts(
        linkAction,
        "-shared",
        "-o",
        analysisMock.getProductName() + "-out/.+/genrule/thebinary.so",
        "-Wl,-whole-archive",
        analysisMock.getProductName() + "-out/.+/genrule/genlib.a",
        "-Wl,-no-whole-archive");
  }

  /**
   * Tests that the shared library version of a cc_library includes linkopts settings
   * in its link command line, but the archive library version doesn't.
   */
  @Test
  public void testCcLibraryLinkopts() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));

    scratch.overwriteFile(
        "custom_malloc/BUILD",
        "cc_library(name = 'custom_malloc',",
        "           srcs = ['custom_malloc.cc'],",
        "           linkopts = ['-Lmalloc_dir -lmalloc_opt']);");

    ConfiguredTarget ccLib = getConfiguredTarget("//custom_malloc:custom_malloc");
    final String linkOpt1 = "-Lmalloc_dir";
    final String linkOpt2 = "-lmalloc_opt";

    // Archive library version:
    Artifact archiveLib =
        Iterables.getOnlyElement(
            Iterables.filter(
                ccLib.getProvider(FileProvider.class).getFilesToBuild().toList(),
                (artifact) -> artifact.getFilename().equals("libcustom_malloc.a")));
    CppLinkAction archiveLink = (CppLinkAction) getGeneratingAction(archiveLib);
    List<String> args = archiveLink.getArguments();
    assertThat(args).doesNotContain(linkOpt1);
    assertThat(args).doesNotContain(linkOpt2);

    // Shared library version:
    Artifact soLib =
        Iterables.getOnlyElement(
            ccLib
                .get(CcInfo.PROVIDER)
                .getCcLinkingContext()
                .getDynamicLibrariesForRuntime(/* linkingStatically= */ false));
    // This artifact is generated by a SolibSymlinkAction, so we need to go back two levels.
    CppLinkAction solibLink =
        (CppLinkAction) getGeneratingAction(getGeneratingAction(soLib).getPrimaryInput());
    args = solibLink.getArguments();
    assertThat(args).contains(linkOpt1);
    assertThat(args).contains(linkOpt2);
  }
}
