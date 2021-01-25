// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link LinkCommandLine}. In particular, tests command line emitted subject to the
 * presence of certain build variables.
 */
@RunWith(JUnit4.class)
public final class LinkCommandLineTest extends BuildViewTestCase {

  private Artifact scratchArtifact(String s) {
    Path execRoot = outputBase.getRelative("exec");
    String outSegment = "root";
    Path outputRoot = execRoot.getRelative(outSegment);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, false, false, false, outSegment);
    try {
      return ActionsTestUtil.createArtifact(
          root, scratch.overwriteFile(outputRoot.getRelative(s).toString()));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private CcToolchainVariables.Builder getMockBuildVariables() {
    return getMockBuildVariables(ImmutableList.<String>of());
  }

  private static CcToolchainVariables.Builder getMockBuildVariables(
      ImmutableList<String> linkstampOutputs) {
    CcToolchainVariables.Builder result = CcToolchainVariables.builder();

    result.addStringVariable(LinkBuildVariables.GENERATE_INTERFACE_LIBRARY.getVariableName(), "no");
    result.addStringVariable(
        LinkBuildVariables.INTERFACE_LIBRARY_INPUT.getVariableName(), "ignored");
    result.addStringVariable(
        LinkBuildVariables.INTERFACE_LIBRARY_OUTPUT.getVariableName(), "ignored");
    result.addStringVariable(
        LinkBuildVariables.INTERFACE_LIBRARY_BUILDER.getVariableName(), "ignored");
    result.addStringSequenceVariable(
        LinkBuildVariables.LINKSTAMP_PATHS.getVariableName(), linkstampOutputs);

    return result;
  }

  private FeatureConfiguration getMockFeatureConfiguration() throws Exception {
    ImmutableList<CToolchain.Feature> features =
        new ImmutableList.Builder<CToolchain.Feature>()
            .addAll(
                CppActionConfigs.getLegacyFeatures(
                    CppPlatform.LINUX,
                    ImmutableSet.of(),
                    "MOCK_LINKER_TOOL",
                    /* supportsEmbeddedRuntimes= */ true,
                    /* supportsInterfaceSharedLibraries= */ false,
                    /* doNotSplitLinkingCmdline= */ true))
            .addAll(
                CppActionConfigs.getFeaturesToAppearLastInFeaturesList(
                    ImmutableSet.of(), /* doNotSplitLinkingCmdline= */ true))
            .build();

    ImmutableList<CToolchain.ActionConfig> actionConfigs =
        CppActionConfigs.getLegacyActionConfigs(
            CppPlatform.LINUX,
            "MOCK_GCC_TOOL",
            "MOCK_AR_TOOL",
            "MOCK_STRIP_TOOL",
            /* supportsInterfaceSharedLibraries= */ false,
            /* existingActionConfigNames= */ ImmutableSet.of());

    return CcToolchainTestHelper.buildFeatures(features, actionConfigs)
        .getFeatureConfiguration(
            ImmutableSet.of(
                Link.LinkTargetType.EXECUTABLE.getActionName(),
                Link.LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName(),
                Link.LinkTargetType.STATIC_LIBRARY.getActionName(),
                CppActionNames.CPP_COMPILE,
                CppActionNames.LINKSTAMP_COMPILE,
                CppRuleClasses.INCLUDES,
                CppRuleClasses.PREPROCESSOR_DEFINES,
                CppRuleClasses.INCLUDE_PATHS,
                CppRuleClasses.PIC));
  }

  private LinkCommandLine.Builder minimalConfiguration(CcToolchainVariables.Builder variables)
      throws Exception {
    return new LinkCommandLine.Builder()
        .setBuildVariables(variables.build())
        .setFeatureConfiguration(getMockFeatureConfiguration());
  }

  private LinkCommandLine.Builder minimalConfiguration() throws Exception {
    return minimalConfiguration(getMockBuildVariables());
  }

  private void assertError(String expectedSubstring, LinkCommandLine.Builder builder) {
    RuntimeException e = assertThrows(RuntimeException.class, () -> builder.build());
    assertThat(e).hasMessageThat().contains(expectedSubstring);
  }

  @Test
  public void testStaticLinkWithBuildInfoHeadersIsError() throws Exception {
    assertError(
        "build info headers may only be present",
        minimalConfiguration()
            .setLinkTargetType(LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .setBuildInfoHeaderArtifacts(
                ImmutableList.of(scratchArtifact("FakeBuildInfoHeaderArtifact1"))));
  }

  /**
   * Tests that when linking without linkstamps, the exec command is the same as the link command.
   */
  @Test
  public void testLinkCommandIsExecCommandWhenNoLinkstamps() throws Exception {
    LinkCommandLine linkConfig =
        minimalConfiguration()
            .setActionName(LinkTargetType.EXECUTABLE.getActionName())
            .setLinkTargetType(LinkTargetType.EXECUTABLE)
            .build();
    List<String> rawLinkArgv = linkConfig.getRawLinkArgv();
    assertThat(linkConfig.arguments()).isEqualTo(rawLinkArgv);
  }

  /** Tests that symbol count output does not appear in argv when it should not. */
  @Test
  public void testSymbolCountsDisabled() throws Exception {
    LinkCommandLine linkConfig =
        minimalConfiguration()
            .forceToolPath("foo/bar/gcc")
            .setLinkTargetType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .build();
    List<String> argv = linkConfig.getRawLinkArgv();
    for (String arg : argv) {
      assertThat(arg).doesNotContain("print-symbol-counts");
    }
  }

  @Test
  public void testLibrariesToLink() throws Exception {
    CcToolchainVariables.Builder variables =
        getMockBuildVariables()
            .addCustomBuiltVariable(
                LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(),
                new SequenceBuilder()
                    .addValue(LibraryToLinkValue.forStaticLibrary("foo", false))
                    .addValue(LibraryToLinkValue.forStaticLibrary("bar", true)));

    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .forceToolPath("foo/bar/gcc")
            .setActionName(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .build();
    String commandLine = Joiner.on(" ").join(linkConfig.getRawLinkArgv());
    assertThat(commandLine).matches(".*foo -Wl,-whole-archive bar -Wl,-no-whole-archive.*");
  }

  @Test
  public void testLibrarySearchDirectories() throws Exception {
    CcToolchainVariables.Builder variables =
        getMockBuildVariables()
            .addStringSequenceVariable(
                LinkBuildVariables.LIBRARY_SEARCH_DIRECTORIES.getVariableName(),
                ImmutableList.of("foo", "bar"));

    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .setActionName(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .build();
    assertThat(linkConfig.getRawLinkArgv()).containsAtLeast("-Lfoo", "-Lbar").inOrder();
  }

  @Test
  public void testLinkerParamFileForStaticLibrary() throws Exception {
    CcToolchainVariables.Builder variables =
        getMockBuildVariables()
            .addStringVariable(
                LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "foo/bar.param");

    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .setActionName(LinkTargetType.STATIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(Link.LinkingMode.STATIC)
            .build();
    assertThat(linkConfig.getRawLinkArgv()).contains("@foo/bar.param");
  }

  @Test
  public void testLinkerParamFileForDynamicLibrary() throws Exception {
    CcToolchainVariables.Builder variables =
        getMockBuildVariables()
            .addStringVariable(
                LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "foo/bar.param");

    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .setActionName(LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .setLinkingMode(Link.LinkingMode.STATIC)
            .doNotSplitLinkingCmdLine()
            .build();
    assertThat(linkConfig.getRawLinkArgv()).contains("@foo/bar.param");
  }

  private List<String> basicArgv(LinkTargetType targetType) throws Exception {
    return basicArgv(targetType, getMockBuildVariables());
  }

  private List<String> basicArgv(LinkTargetType targetType, CcToolchainVariables.Builder variables)
      throws Exception {
    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .setActionName(targetType.getActionName())
            .setLinkTargetType(targetType)
            .setLinkingMode(LinkingMode.STATIC)
            .build();
    return linkConfig.arguments();
  }

  /** Tests that a "--force_pic" configuration applies "-pie" to executable links. */
  @Test
  public void testPicMode() throws Exception {
    String pieArg = "-pie";

    // Disabled:
    assertThat(basicArgv(LinkTargetType.EXECUTABLE)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.STATIC_LIBRARY)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.PIC_STATIC_LIBRARY)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY)).doesNotContain(pieArg);

    CcToolchainVariables.Builder picVariables =
        getMockBuildVariables()
            .addStringVariable(LinkBuildVariables.FORCE_PIC.getVariableName(), "");
    // Enabled:
    useConfiguration("--force_pic");
    assertThat(basicArgv(LinkTargetType.EXECUTABLE, picVariables)).contains(pieArg);
    assertThat(basicArgv(LinkTargetType.NODEPS_DYNAMIC_LIBRARY, picVariables))
        .doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.STATIC_LIBRARY, picVariables)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.PIC_STATIC_LIBRARY, picVariables)).doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY, picVariables))
        .doesNotContain(pieArg);
    assertThat(basicArgv(LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY, picVariables))
        .doesNotContain(pieArg);
  }

  @Test
  public void testSplitStaticLinkCommand() throws Exception {
    useConfiguration("--nostart_end_lib");
    Artifact paramFile = scratchArtifact("some/file.params");
    LinkCommandLine linkConfig =
        minimalConfiguration(
                getMockBuildVariables()
                    .addStringVariable(
                        LinkBuildVariables.OUTPUT_EXECPATH.getVariableName(), "a/FakeOutput")
                    .addStringVariable(
                        LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "some/file.params"))
            .setActionName(LinkTargetType.STATIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.STATIC_LIBRARY)
            .forceToolPath("foo/bar/ar")
            .setParamFile(paramFile)
            .build();
    Pair<List<String>, List<String>> result = linkConfig.splitCommandline();
    assertThat(result.first).isEqualTo(Arrays.asList("foo/bar/ar", "@some/file.params"));
    assertThat(result.second).isEqualTo(Arrays.asList("rcsD", "a/FakeOutput"));
  }

  @Test
  public void testSplitDynamicLinkCommand() throws Exception {
    useConfiguration("--nostart_end_lib");
    Artifact paramFile = scratchArtifact("some/file.params");
    LinkCommandLine linkConfig =
        minimalConfiguration(
                getMockBuildVariables()
                    .addStringVariable(
                        LinkBuildVariables.OUTPUT_EXECPATH.getVariableName(), "a/FakeOutput")
                    .addStringVariable(
                        LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "some/file.params")
                    .addStringSequenceVariable(
                        LinkBuildVariables.USER_LINK_FLAGS.getVariableName(), ImmutableList.of("")))
            .setActionName(LinkTargetType.DYNAMIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.DYNAMIC_LIBRARY)
            .forceToolPath("foo/bar/linker")
            .setParamFile(paramFile)
            .doNotSplitLinkingCmdLine()
            .build();
    Pair<List<String>, List<String>> result = linkConfig.splitCommandline();
    assertThat(result.first).containsExactly("foo/bar/linker", "@some/file.params").inOrder();
    assertThat(result.second).containsExactly("-shared", "-o", "a/FakeOutput", "").inOrder();
  }

  @Test
  public void testStaticLinkCommand() throws Exception {
    useConfiguration("--nostart_end_lib");
    LinkCommandLine linkConfig =
        minimalConfiguration(
                getMockBuildVariables()
                    .addStringVariable(
                        LinkBuildVariables.OUTPUT_EXECPATH.getVariableName(), "a/FakeOutput"))
            .forceToolPath("foo/bar/ar")
            .setActionName(LinkTargetType.STATIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.STATIC_LIBRARY)
            .build();
    List<String> result = linkConfig.getRawLinkArgv();
    assertThat(result).isEqualTo(Arrays.asList("foo/bar/ar", "rcsD", "a/FakeOutput"));
  }

  @Test
  public void testSplitAlwaysLinkLinkCommand() throws Exception {
    CcToolchainVariables.Builder variables =
        CcToolchainVariables.builder()
            .addStringVariable(CcCommon.SYSROOT_VARIABLE_NAME, "/usr/grte/v1")
            .addStringVariable(LinkBuildVariables.OUTPUT_EXECPATH.getVariableName(), "a/FakeOutput")
            .addStringVariable(
                LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "some/file.params")
            .addCustomBuiltVariable(
                LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(),
                new CcToolchainVariables.SequenceBuilder()
                    .addValue(LibraryToLinkValue.forObjectFile("foo.o", false))
                    .addValue(LibraryToLinkValue.forObjectFile("bar.o", false)));

    Artifact paramFile = scratchArtifact("some/file.params");
    LinkCommandLine linkConfig =
        minimalConfiguration(variables)
            .setActionName(LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
            .forceToolPath("foo/bar/ar")
            .setParamFile(paramFile)
            .build();
    Pair<List<String>, List<String>> result = linkConfig.splitCommandline();

    assertThat(result.first).isEqualTo(Arrays.asList("foo/bar/ar", "@some/file.params"));
    assertThat(result.second).isEqualTo(Arrays.asList("rcsD", "a/FakeOutput", "foo.o", "bar.o"));
  }

  private SpecialArtifact createTreeArtifact(String name) {
    FileSystem fs = scratch.getFileSystem();
    Path execRoot = fs.getPath(TestUtils.tmpDir());
    PathFragment execPath = PathFragment.create("out").getRelative(name);
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        ArtifactRoot.asDerivedRoot(execRoot, false, false, false, "out"), execPath);
  }

  private void verifyArguments(
      Iterable<String> arguments,
      Iterable<String> allowedArguments,
      Iterable<String> disallowedArguments) {
    assertThat(arguments).containsAtLeastElementsIn(allowedArguments);
    assertThat(arguments).containsNoneIn(disallowedArguments);
  }

  @Test
  public void testTreeArtifactLink() throws Exception {
    SpecialArtifact testTreeArtifact = createTreeArtifact("library_directory");

    TreeFileArtifact library0 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library1.o");

    ArtifactExpander expander =
        new ArtifactExpander() {
          @Override
          public void expand(Artifact artifact, Collection<? super Artifact> output) {
            if (artifact.equals(testTreeArtifact)) {
              output.add(library0);
              output.add(library1);
            }
          };
        };

    Iterable<String> treeArtifactsPaths = ImmutableList.of(testTreeArtifact.getExecPathString());
    Iterable<String> treeFileArtifactsPaths =
        ImmutableList.of(library0.getExecPathString(), library1.getExecPathString());

    Artifact paramFile = scratchArtifact("some/file.params");

    LinkCommandLine linkConfig =
        minimalConfiguration(
                getMockBuildVariables()
                    .addStringVariable(
                        LinkBuildVariables.LINKER_PARAM_FILE.getVariableName(), "some/file.params")
                    .addCustomBuiltVariable(
                        LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(),
                        new CcToolchainVariables.SequenceBuilder()
                            .addValue(
                                LibraryToLinkValue.forObjectFileGroup(
                                    ImmutableList.of(testTreeArtifact), false))))
            .forceToolPath("foo/bar/gcc")
            .setActionName(LinkTargetType.STATIC_LIBRARY.getActionName())
            .setLinkTargetType(LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(Link.LinkingMode.STATIC)
            .setParamFile(paramFile)
            .build();

    // Should only reference the tree artifact.
    verifyArguments(linkConfig.arguments(null), treeArtifactsPaths, treeFileArtifactsPaths);
    verifyArguments(linkConfig.getRawLinkArgv(null), treeArtifactsPaths, treeFileArtifactsPaths);
    verifyArguments(
        linkConfig.paramCmdLine().arguments(null), treeArtifactsPaths, treeFileArtifactsPaths);

    // Should only reference tree file artifacts.
    verifyArguments(linkConfig.arguments(expander), treeFileArtifactsPaths, treeArtifactsPaths);
    verifyArguments(
        linkConfig.getRawLinkArgv(expander), treeFileArtifactsPaths, treeArtifactsPaths);
    verifyArguments(
        linkConfig.paramCmdLine().arguments(expander), treeFileArtifactsPaths, treeArtifactsPaths);
  }
}
