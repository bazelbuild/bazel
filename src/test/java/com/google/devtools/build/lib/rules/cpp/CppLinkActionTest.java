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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CommandLineLimits;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.CommandLines.ExpandedCommandLines;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.ResourceSetOrBuilder;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.LegacyLinkerInputs.LibraryInput;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.EnvEntry;
import java.io.IOException;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CppLinkAction}. */
@RunWith(JUnit4.class)
public final class CppLinkActionTest extends BuildViewTestCase {

  @Before
  public void setupCcToolchainConfig() throws IOException {
    scratch.overwriteFile(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
    scratch.appendFile("tools/cpp/BUILD");
  }

  private RuleContext createDummyRuleContext() throws Exception {
    return view.getRuleContextForTesting(
        reporter,
        scratchConfiguredTarget(
            "dummyRuleContext",
            "dummyRuleContext",
            // CppLinkAction creation requires a CcToolchainProvider.
            "cc_library(name = 'dummyRuleContext')"),
        new StubAnalysisEnvironment() {
          @Override
          public void registerAction(ActionAnalysisMetadata action) {
            // No-op.
          }

          @Override
          public StarlarkSemantics getStarlarkSemantics() {
            return StarlarkSemantics.DEFAULT;
          }

          @Override
          public Artifact.DerivedArtifact getDerivedArtifact(
              PathFragment rootRelativePath, ArtifactRoot root) {
            return CppLinkActionTest.this.getDerivedArtifact(
                rootRelativePath, root, ActionsTestUtil.NULL_ARTIFACT_OWNER);
          }

          @Override
          public Artifact.DerivedArtifact getDerivedArtifact(
              PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPaths) {
            Preconditions.checkArgument(
                !contentBasedPaths, "C++ tests don't use content-based outputs");
            return getDerivedArtifact(rootRelativePath, root);
          }
        });
  }

  private static FeatureConfiguration getMockFeatureConfiguration(
      ImmutableMap<String, String> envVars) {
    CToolchain.FlagGroup flagGroup =
        CToolchain.FlagGroup.newBuilder().addFlag("-lcpp_standard_library").build();
    CToolchain.FlagSet flagSet =
        CToolchain.FlagSet.newBuilder()
            .addAction("c++-link-executable")
            .addFlagGroup(flagGroup)
            .build();
    CToolchain.EnvSet.Builder envSet =
        CToolchain.EnvSet.newBuilder()
            .addAction("c++-link-executable")
            .addAction("c++-link-static-library")
            .addAction("c++-link-dynamic-library")
            .addAction("c++-link-nodeps-dynamic-library");
    envVars.forEach(
        (var, val) -> envSet.addEnvEntry(EnvEntry.newBuilder().setKey(var).setValue(val).build()));

    CToolchain.Feature linkCppStandardLibrary =
        CToolchain.Feature.newBuilder()
            .setName("link_cpp_standard_library")
            .setEnabled(true)
            .addFlagSet(flagSet)
            .addEnvSet(envSet.build())
            .build();
    CToolchain.Feature archiveParamFile =
        CToolchain.Feature.newBuilder()
            .setName(CppRuleClasses.ARCHIVE_PARAM_FILE)
            .setEnabled(true)
            .build();
    ImmutableList<CToolchain.Feature> features =
        new ImmutableList.Builder<CToolchain.Feature>()
            .addAll(
                CppActionConfigs.getLegacyFeatures(
                    CppPlatform.LINUX,
                    ImmutableSet.of(),
                    "dynamic_library_linker_tool",
                    /* supportsEmbeddedRuntimes= */ true,
                    /* supportsInterfaceSharedLibraries= */ false))
            .addAll(CppActionConfigs.getFeaturesToAppearLastInFeaturesList(ImmutableSet.of()))
            .add(linkCppStandardLibrary)
            .add(archiveParamFile)
            .build();

    ImmutableList<CToolchain.ActionConfig> actionConfigs =
        CppActionConfigs.getLegacyActionConfigs(
            CppPlatform.LINUX,
            "gcc_tool",
            "ar_tool",
            "strip_tool",
            /* supportsInterfaceSharedLibraries= */ false,
            /* existingActionConfigNames= */ ImmutableSet.of());

    try {
      return CcToolchainTestHelper.buildFeatures(features, actionConfigs)
          .getFeatureConfiguration(
              ImmutableSet.of(
                  "link_cpp_standard_library",
                  CppRuleClasses.ARCHIVE_PARAM_FILE,
                  LinkTargetType.EXECUTABLE.getActionName(),
                  LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName(),
                  LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
                  LinkTargetType.STATIC_LIBRARY.getActionName()));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public void registerToolchainWithConfig(String... config) throws IOException {
    scratch.file(
        "toolchain/crosstool_rule.bzl",
        """
        load(
            "//tools/cpp:cc_toolchain_config_lib.bzl",
            "action_config",
            "env_entry",
            "env_set",
            "feature",
            "feature_set",
            "flag_group",
            "flag_set",
            "tool",
            "tool_path",
        )

        def _impl(ctx):
            return cc_common.create_cc_toolchain_config_info(
                ctx = ctx,
                toolchain_identifier = "",
                compiler = "",
        """,
        String.join("\\n", config),
        """
            )

        cc_toolchain_config_rule = rule(
            implementation = _impl,
            attrs = {},
            provides = [CcToolchainConfigInfo],
            fragments = ["cpp"],
        )
        """);
    scratch.file(
        "toolchain/BUILD",
        """
load(":crosstool_rule.bzl", "cc_toolchain_config_rule")
cc_toolchain_config_rule(name = "toolchain_config")
filegroup(name = "empty")
cc_toolchain(
    name = "cc_toolchain",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    toolchain_config = ":toolchain_config",
)
toolchain(name = "toolchain", toolchain = ":cc_toolchain", toolchain_type = '\
"""
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/cpp:toolchain_type')");
  }

  @Test
  public void testToolchainFeatureFlags() throws Exception {
    registerToolchainWithConfig(
        """
        features = [feature(
            name = "a",
            flag_sets = [flag_set(
                actions = ["c++-link-executable"],
                flag_groups = [flag_group(flags = ["some_flag"])],
            )],
        )]
        """);
    useConfiguration("--features=a", "--extra_toolchains=//toolchain");
    scratch.file("foo/BUILD", "cc_binary(name = 'foo')");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));
    assertThat(linkAction.getArguments()).contains("some_flag");
  }

  @Test
  public void testExecutionRequirementsFromCrosstool() throws Exception {
    registerToolchainWithConfig(
        """
        action_configs = [action_config(
            action_name = "c++-link-executable",
            tools = [tool(
                path = "DUMMY_TOOL",
                execution_requirements = ["dummy-exec-requirement"],
            )],
        )]
        """);
    useConfiguration("--extra_toolchains=//toolchain");
    scratch.file("foo/BUILD", "cc_binary(name = 'foo')");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));
    assertThat(linkAction.getExecutionInfo()).containsEntry("dummy-exec-requirement", "");
  }

  @Test
  public void testLibOptsAndLibSrcsAreInCorrectOrder() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = [
                "some-dir/libbar.so",
                "some-other-dir/qux.so",
            ],
            linkopts = [
                "-ldl",
                "-lutil",
            ],
        )
        """);
    scratch.file("x/some-dir/libbar.so");
    scratch.file("x/some-other-dir/qux.so");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:foo");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/foo");

    List<String> arguments = linkAction.getArguments();

    assertThat(Joiner.on(" ").join(arguments))
        .matches(
            ".* -L[^ ]*some-dir(?= ).* -L[^ ]*some-other-dir(?= ).* "
                + "-lbar -l:qux.so(?= ).* -ldl -lutil .*");
    assertThat(Joiner.on(" ").join(arguments))
        .matches(
            ".* -Xlinker -rpath -Xlinker [^ ]*some-dir(?= ).* -Xlinker -rpath -Xlinker [^"
                + " ]*some-other-dir .*");
  }

  @Test
  public void testLegacyWholeArchiveHasNoEffectOnDynamicModeDynamicLibraries() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "libfoo.so",
            srcs = ["foo.cc"],
            linkshared = 1,
            linkstatic = 0,
        )
        """);
    useConfiguration("--legacy_whole_archive");
    assertThat(getLibfooArguments()).doesNotContain("-Wl,-whole-archive");
  }

  private List<String> getLibfooArguments() throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:libfoo.so");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/libfoo.so");
    return linkAction.getArguments();
  }

  @Test
  public void testExposesRuntimeLibrarySearchDirectoriesVariable() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = [
                "some-dir/bar.so",
                "some-other-dir/qux.so",
            ],
        )
        """);
    scratch.file("x/some-dir/bar.so");
    scratch.file("x/some-other-dir/qux.so");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:foo");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/foo");

    assertThat(Joiner.on(" ").join(linkAction.getArguments()))
        .matches(".*some-dir .*some-other-dir.*");
  }

  @Test
  public void testCompilesDynamicModeTestSourcesWithFeatureIntoDynamicLibrary() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Skip the test on Windows.
      // TODO(#7524): This test should work on Windows just fine, investigate and fix.
      return;
    }
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.SUPPORTS_PIC,
                    CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                    CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES));
    scratch.file(
        "x/BUILD",
        """
        cc_test(
            name = "a",
            srcs = ["a.cc"],
            features = ["dynamic_link_test_srcs"],
        )

        cc_binary(
            name = "b",
            srcs = ["a.cc"],
        )

        cc_test(
            name = "c",
            srcs = ["a.cc"],
            features = ["dynamic_link_test_srcs"],
            linkstatic = 1,
        )
        """);
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--force_pic");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin _solib_k8/libx_Sliba.ifso");
    assertThat(linkAction.getArguments())
        .contains(getBinArtifactWithNoOwner("_solib_k8/libx_Sliba.ifso").getExecPathString());
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .contains("bin _solib_k8/libx_Sliba.so");

    configuredTarget = getConfiguredTarget("//x:b");
    linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/b");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/b/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/b");

    configuredTarget = getConfiguredTarget("//x:c");
    linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/c");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/c/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/c");
  }

  @Test
  public void testCompilesDynamicModeBinarySourcesWithoutFeatureIntoDynamicLibrary()
      throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Skip the test on Windows.
      // TODO(#7524): This test should work on Windows just fine, investigate and fix.
      return;
    }
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER, CppRuleClasses.SUPPORTS_PIC));
    scratch.file(
        "x/BUILD", "cc_binary(name = 'a', srcs = ['a.cc'], features = ['-static_link_test_srcs'])");
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--force_pic", "--dynamic_mode=default");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .doesNotContain("bin _solib_k8/libx_Sliba.ifso");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/a/a.pic.o");
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/a");
  }

  @Test
  public void testToolchainFeatureEnv() throws Exception {
    registerToolchainWithConfig(
        """
        features = [feature(
            name = "a",
            env_sets = [env_set(
                actions = ["c++-link-executable"],
                env_entries = [env_entry(key = "foo", value = "bar")],
            )],
        )]
        """);
    useConfiguration("--features=a", "--extra_toolchains=//toolchain");
    scratch.file("foo/BUILD", "cc_binary(name = 'foo')");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));
    assertThat(linkAction.getEffectiveEnvironment(ImmutableMap.of())).containsEntry("foo", "bar");
  }

  private enum StaticKeyAttributes {
    OUTPUT_FILE,
    ENVIRONMENT,
  }

  /**
   * This mainly checks that static library links don't have identical keys, and it also compares
   * them with simple dynamic library links.
   */
  @Test
  public void testComputeKeyStatic() throws Exception {
    final RuleContext ruleContext = createDummyRuleContext();
    final PathFragment staticOutputPath = PathFragment.create("dummyRuleContext/output/path.a");
    final PathFragment dynamicOutputPath = PathFragment.create("dummyRuleContext/output/path.so");
    final Artifact staticOutputFile = getBinArtifactWithNoOwner(staticOutputPath.getPathString());
    final Artifact dynamicOutputFile = getBinArtifactWithNoOwner(dynamicOutputPath.getPathString());

    ActionTester.runTest(
        StaticKeyAttributes.class,
        new ActionCombinationFactory<StaticKeyAttributes>() {

          @Override
          public Action generate(ImmutableSet<StaticKeyAttributes> attributes)
              throws RuleErrorException {
            try {
              CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
              CppLinkActionBuilder builder =
                  new CppLinkActionBuilder(
                      CppLinkActionBuilder.newActionConstruction(ruleContext),
                      attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                          ? staticOutputFile
                          : dynamicOutputFile,
                      toolchain,
                      toolchain.getFdoContext(),
                      getMockFeatureConfiguration(
                          attributes.contains(StaticKeyAttributes.ENVIRONMENT)
                              ? ImmutableMap.of("var", "value")
                              : ImmutableMap.of()),
                      MockCppSemantics.INSTANCE);
              builder.setLinkType(
                  attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                      ? LinkTargetType.STATIC_LIBRARY
                      : LinkTargetType.NODEPS_DYNAMIC_LIBRARY);
              builder.setLibraryIdentifier("foo");
              return builder.build();
            } catch (EvalException | InterruptedException e) {
              throw new RuleErrorException(e.getMessage());
            }
          }
        },
        actionKeyContext);
  }

  @Test
  public void testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForCcLibrary()
      throws Exception {
    useConfiguration("--features=-archive_param_file");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction linkAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));

    assertThat(getCommandLine(linkAction).paramFileInfo).isNull();
  }

  @Test
  public void testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForObjcLibrary()
      throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    useConfiguration(
        "--features=-archive_param_file", "--platforms=" + MockObjcSupport.DARWIN_X86_64);
    invalidatePackages();
    scratch.file("foo/BUILD", "objc_library(name = 'foo', srcs = ['foo.m'])");

    SpawnAction linkAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));

    assertThat(getCommandLine(linkAction).paramFileInfo).isNull();
  }

  @Test
  public void testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForIfSo()
      throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                    CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES));
    useConfiguration();
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));
    assertThat(getCommandLine(linkAction).paramFileInfo).isNull();

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    assertThat(getCommandLine(archiveAction).paramFileInfo).isNull();
  }

  @Test
  public void
      testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForPicStaticLibrary()
          throws Exception {
    useConfiguration("--features=-archive_param_file", "--force_pic");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    assertThat(getCommandLine(archiveAction).paramFileInfo).isNull();
  }

  @Test
  public void
      testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForAlwayslinkStaticLibrary()
          throws Exception {
    useConfiguration("--features=-archive_param_file");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], alwayslink = True)");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    assertThat(getCommandLine(archiveAction).paramFileInfo).isNull();
  }

  @Test
  public void
      testCommandLineSplittingWithoutArchiveParamFileFeature_shouldBeOffForAlwayslinkPicStaticLibrary()
          throws Exception {
    useConfiguration("--features=*archive_param_file", "--force_pic");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], alwayslink = True)");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    assertThat(getCommandLine(archiveAction).paramFileInfo).isNull();
  }

  @Test
  public void testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForCcLibrary()
      throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction linkAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));

    var commandLine = getCommandLine(linkAction);
    assertThat(commandLine.paramFileInfo).isNotNull();
    assertThat(commandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOffForIfSo()
      throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                    CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES,
                    CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file,supports_dynamic_linker");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));
    assertThat(getCommandLine(linkAction).paramFileInfo).isNull();

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    assertThat(getCommandLine(archiveAction).paramFileInfo).isNull();
  }

  @Test
  public void testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForStaticLibrary()
      throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    var archiveCommandLine = getCommandLine(archiveAction);
    assertThat(archiveCommandLine.paramFileInfo).isNotNull();
    assertThat(archiveCommandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForPicStaticLibrary()
      throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file", "--force_pic");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    var archiveCommandLine = getCommandLine(archiveAction);
    assertThat(archiveCommandLine.paramFileInfo).isNotNull();
    assertThat(archiveCommandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void
      testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForAlwayslinkStaticLibrary()
          throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], alwayslink = True)");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    var archiveCommandLine = getCommandLine(archiveAction);
    assertThat(archiveCommandLine.paramFileInfo).isNotNull();
    assertThat(archiveCommandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void
      testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForAlwayslinkPicStaticLibrary()
          throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    useConfiguration("--features=archive_param_file", "--force_pic");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], alwayslink = True)");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    var archiveCommandLine = getCommandLine(archiveAction);
    assertThat(archiveCommandLine.paramFileInfo).isNotNull();
    assertThat(archiveCommandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void testCommandLineSplittingWithArchiveParamFileFeature_shouldBeOnForObjcLibrary()
      throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    MockObjcSupport.setupCcToolchainConfig(
        mockToolsConfig,
        MockObjcSupport.darwinX86_64().withFeatures(CppRuleClasses.ARCHIVE_PARAM_FILE));
    invalidatePackages();
    useConfiguration(
        "--features=archive_param_file",
        "--platforms=" + MockObjcSupport.DARWIN_X86_64,
        "--experimental_starlark_linking");
    scratch.file("foo/BUILD", "objc_library(name = 'foo', srcs = ['foo.m'])");

    SpawnAction archiveAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));
    var archiveCommandLine = getCommandLine(archiveAction);
    assertThat(archiveCommandLine.paramFileInfo).isNotNull();
    assertThat(archiveCommandLine.paramFileInfo.always()).isTrue();
  }

  @Test
  public void testLocalLinkResourceEstimate() throws Exception {
    scratch.file("foo/BUILD", "cc_binary(name = 'foo')");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    ResourceSetOrBuilder resourceSetOrBuilder = linkAction.getResourceSetOrBuilder();
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.DARWIN, 100))
        .isEqualTo(ResourceSet.createWithRamCpu(20, 1));
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.DARWIN, 1000))
        .isEqualTo(ResourceSet.createWithRamCpu(65, 1));
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.LINUX, 100))
        .isEqualTo(ResourceSet.createWithRamCpu(50, 1));
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.LINUX, 10000))
        .isEqualTo(ResourceSet.createWithRamCpu(900, 1));
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.WINDOWS, 0))
        .isEqualTo(ResourceSet.createWithRamCpu(1500, 1));
    assertThat(resourceSetOrBuilder.buildResourceSet(OS.WINDOWS, 1000))
        .isEqualTo(ResourceSet.createWithRamCpu(2500, 1));
  }

  private static CppLinkActionBuilder createLinkBuilder(
      RuleContext ruleContext,
      Link.LinkTargetType type,
      String outputPath,
      Iterable<Artifact> nonLibraryInputs,
      ImmutableList<LibraryInput> libraryInputs,
      FeatureConfiguration featureConfiguration)
      throws RuleErrorException, EvalException {
    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    return new CppLinkActionBuilder(
            CppLinkActionBuilder.newActionConstruction(ruleContext),
            ActionsTestUtil.createArtifact(ruleContext.getBinDirectory(), outputPath),
            toolchain,
            toolchain.getFdoContext(),
            featureConfiguration,
            MockCppSemantics.INSTANCE)
        .addObjectFiles(nonLibraryInputs)
        .addLibraries(libraryInputs)
        .setLinkType(type)
        .setLinkerFiles(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
        .setLinkingMode(LinkingMode.STATIC);
  }

  private static CppLinkActionBuilder createLinkBuilder(
      RuleContext ruleContext, Link.LinkTargetType type) throws Exception {
    PathFragment output = PathFragment.create("dummyRuleContext/output/path.a");
    return createLinkBuilder(
        ruleContext,
        type,
        output.getPathString(),
        ImmutableList.of(),
        ImmutableList.of(),
        getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()));
  }

  private Artifact scratchArtifact(String s) {
    Path execRoot = outputBase.getRelative("exec");
    String outSegment = "out";
    Path outputRoot = execRoot.getRelative(outSegment);
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, outSegment);
    try {
      return ActionsTestUtil.createArtifact(
          root, scratch.overwriteFile(outputRoot.getRelative(s).toString()));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void assertError(String expectedSubstring, CppLinkActionBuilder builder) {
    Exception e = assertThrows(Exception.class, builder::build);
    assertThat(e).hasMessageThat().contains(expectedSubstring);
  }

  @Test
  public void testInterfaceOutputWithoutBuildingDynamicLibraryIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.EXECUTABLE)
            .setInterfaceOutput(scratchArtifact("FakeInterfaceOutput"));

    assertError("Interface output can only be used with DYNAMIC_LIBRARY targets", builder);
  }

  @Test
  public void testInterfaceOutputForDynamicLibrary() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.SUPPORTS_DYNAMIC_LINKER,
                    CppRuleClasses.SUPPORTS_INTERFACE_SHARED_LIBRARIES));
    useConfiguration();

    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'])");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:foo");
    assertThat(configuredTarget).isNotNull();
    ImmutableList<String> inputs =
        getGeneratingAction(configuredTarget, "foo/libfoo.so").getInputs().toList().stream()
            .map(Artifact::getExecPathString)
            .collect(ImmutableList.toImmutableList());
    assertThat(inputs.stream().anyMatch(i -> i.contains("tools/cpp/link_dynamic_library")))
        .isTrue();
  }

  @Test
  public void testInterfaceOutputForDynamicLibraryLegacy() throws Exception {
    registerToolchainWithConfig(
        """
features = [
    feature(name = "supports_dynamic_linker", enabled = True),
    feature(name = "supports_interface_shared_libraries", enabled = True),
    feature(
        name = "build_interface_libraries",
        flag_sets = [flag_set(
            actions = ["c++-link-nodeps-dynamic-library"],
            flag_groups = [flag_group(flags = [
                "%{generate_interface_library}",
                "%{interface_library_builder_path}",
                "%{interface_library_input_path}",
                "%{interface_library_output_path}",
            ])],
        )],
    ),
    feature(
        name = "dynamic_library_linker_tool",
        flag_sets = [flag_set(
            actions = ["c++-link-nodeps-dynamic-library"],
            flag_groups = [flag_group(flags = ["dynamic_library_linker_tool"])],
        )],
    ),
    feature(name = "has_configured_linker_path"),
],
action_configs = [action_config(
    action_name = "c++-link-nodeps-dynamic-library",
    tools = [tool(
        path = "custom/crosstool/scripts/link_dynamic_library.sh",
    )],
    implies = ["has_configured_linker_path", "build_interface_libraries", "dynamic_library_linker_tool"],
)]
""");
    useConfiguration(
        "--extra_toolchains=//toolchain",
        "--features=build_interface_libraries,dynamic_library_linker_tool");
    scratch.file("foo/BUILD", "cc_library(name = 'foo', srcs = ['a.c'])");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    List<String> commandLine = linkAction.getArguments();
    assertThat(commandLine).hasSize(12);
    assertThat(commandLine.get(0)).endsWith("custom/crosstool/scripts/link_dynamic_library.sh");
    assertThat(commandLine.get(7)).isEqualTo("yes");
    assertThat(commandLine.get(8)).endsWith("tools/cpp/build_interface_so");
    assertThat(commandLine.get(9)).endsWith("foo.so");
    assertThat(commandLine.get(10)).endsWith("bin/foo/libfoo.ifso");
    assertThat(commandLine.get(11)).isEqualTo("dynamic_library_linker_tool");
  }

  @Test
  public void testStaticLinkWithDynamicIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(Link.LinkingMode.DYNAMIC)
            .setLibraryIdentifier("foo");

    assertError("static library link must be static", builder);
  }

  @Test
  public void testStaticLinkWithNativeDepsIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .setLibraryIdentifier("foo")
            .setNativeDeps(true);

    assertError("the native deps flag must be false for static links", builder);
  }

  @Test
  public void testStaticLinkWithWholeArchiveIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .setLibraryIdentifier("foo")
            .setWholeArchive(true);

    assertError("the need whole archive flag must be false for static links", builder);
  }

  private SpecialArtifact createTreeArtifact(PathFragment execPath) {
    FileSystem fs = scratch.getFileSystem();
    Path execRoot = fs.getPath(TestUtils.tmpDir());
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, execPath), execPath);
  }

  private static void verifyArguments(
      Iterable<String> arguments,
      Iterable<String> allowedArguments,
      Iterable<String> disallowedArguments) {
    assertThat(arguments).containsAtLeastElementsIn(allowedArguments);
    assertThat(arguments).containsNoneIn(disallowedArguments);
  }

  @Test
  public void testLinksTreeArtifactLibraries() throws Exception {
    scratch.file("bazel_internal/test_rules/cc/BUILD");
    scratch.file(
        "bazel_internal/test_rules/cc/foo.bzl",
        """
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
def _impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
    )
    dir = ctx.actions.declare_directory("library_directory")
    ctx.actions.run(executable = ctx.executable._tool, outputs = [dir])
    compilation_outputs = cc_common.create_compilation_outputs(objects = depset([dir]))
    cc_common.link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        name = ctx.label.name,
        compilation_outputs = compilation_outputs
    )

cc_link_rule = rule(
    implementation = _impl,
    attrs = {
        "_tool": attr.label(default = "//foo:tool", executable = True, cfg = "exec"),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)
""");
    scratch.file(
        "foo/BUILD",
        """
load("//bazel_internal/test_rules/cc:foo.bzl", "cc_link_rule")
cc_link_rule(name = "foo")
cc_binary(name = "tool")
""");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    SpecialArtifact testTreeArtifact =
        createTreeArtifact(
            linkAction
                .getPrimaryOutput()
                .getExecPath()
                .getParentDirectory()
                .getChild("library_directory"));
    TreeFileArtifact library0 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library1.o");
    ArtifactExpander expander =
        artifact ->
            artifact.getExecPathString().equals(testTreeArtifact.getExecPathString())
                ? ImmutableSortedSet.of(library0, library1)
                : ImmutableSortedSet.of();

    ImmutableList<String> treeArtifactsPaths =
        ImmutableList.of(testTreeArtifact.getExecPathString());
    ImmutableList<String> treeFileArtifactsPaths =
        ImmutableList.of(library0.getExecPathString(), library1.getExecPathString());

    // Should only reference the tree artifact.
    verifyArguments(linkAction.getArguments(), treeArtifactsPaths, treeFileArtifactsPaths);

    // Should only reference tree file artifacts.
    ExpandedCommandLines expandedCommandLines =
        linkAction
            .getCommandLines()
            .expand(
                expander,
                linkAction.getPrimaryOutput().getExecPath(),
                PathMapper.NOOP,
                CommandLineLimits.UNLIMITED);
    verifyArguments(
        expandedCommandLines.getParamFiles().get(0).getArguments(),
        treeFileArtifactsPaths,
        treeArtifactsPaths);
  }

  /** Tests that -pie is removed when -shared is also present (http://b/5611891#). */
  @Test
  public void testPieOptionDisabledForSharedLibraries() throws Exception {
    scratch.file(
        "foo/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'], linkopts = ['-pie', '-other', '-pie'],"
            + " linkshared = True)");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    List<String> argv = linkAction.getArguments();
    assertThat(argv).doesNotContain("-pie");
    assertThat(argv).contains("-other");
  }

  /** Tests that -pie is kept when -shared is not present (http://b/5611891#). */
  @Test
  public void testPieOptionKeptForExecutables() throws Exception {
    scratch.file(
        "foo/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'], linkopts = ['-pie', '-other', '-pie'],"
            + " linkshared = False)");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    List<String> argv = linkAction.getArguments();
    assertThat(argv).contains("-pie");
    assertThat(argv).contains("-other");
  }

  @Test
  public void testLinkoptsComeAfterLinkerInputs() throws Exception {
    scratch.file(
        "foo/BUILD",
        "cc_library(name = 'bar1', srcs = ['bar.cc'])",
        "cc_library(name = 'bar2', srcs = ['bar.cc'])",
        "cc_binary(name = 'foo', srcs = ['foo.cc'], deps = [':bar1', ':bar2'], linkopts ="
            + " ['FakeLinkopt1', 'FakeLinkopt2'])");

    SpawnAction linkAction = (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppLink"));

    Artifact linkerInput1 = linkAction.getInputs().toList().get(0);
    Artifact linkerInput2 = linkAction.getInputs().toList().get(1);
    Artifact linkerInput3 = linkAction.getInputs().toList().get(2);
    List<String> argv = linkAction.getArguments();
    assertThat(argv)
        .containsAtLeast(
            linkerInput1.getExecPathString(),
            linkerInput2.getExecPathString(),
            linkerInput3.getExecPathString(),
            "FakeLinkopt1",
            "FakeLinkopt2");
    int lastLinkerInputIndex =
        Ints.max(
            argv.indexOf(linkerInput1.getExecPathString()),
            argv.indexOf(linkerInput2.getExecPathString()),
            argv.indexOf(linkerInput3.getExecPathString()));
    int firstLinkoptIndex = Math.min(argv.indexOf("FakeLinkopt1"), argv.indexOf("FakeLinkopt2"));
    assertThat(lastLinkerInputIndex).isLessThan(firstLinkoptIndex);
  }

  @Test
  public void testLinkoptsAreOmittedForStaticLibrary() throws Exception {
    scratch.file(
        "foo/BUILD", "cc_library(name = 'foo', srcs = ['foo.cc'], linkopts = ['FakeLinkopt1'])");

    SpawnAction linkAction =
        (SpawnAction) Iterables.getOnlyElement(getActions("//foo", "CppArchive"));

    assertThat(linkAction.getArguments()).doesNotContain("FakeLinkopt1");
  }

  @Test
  public void testExposesLinkstampObjects() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "bin",
            deps = [":lib"],
        )

        cc_library(
            name = "lib",
            linkstamp = "linkstamp.cc",
        )
        """);
    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:bin");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "x/bin");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin x/_objs/bin/x/linkstamp.o");
  }

  @Test
  public void testGccQuotingForParamFilesFeature_enablesGccQuoting() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.GCC_QUOTING_FOR_PARAM_FILES));
    useConfiguration();

    scratch.file(
        "foo/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = [
                'quote".cc',
                "space .cc",
            ],
        )
        """);
    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:foo");
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(configuredTarget, "foo/foo");

    assertThat(getCommandLine(linkAction).paramFileInfo.getFileType())
        .isEqualTo(ParameterFileType.GCC_QUOTED);
  }

  private CommandLineAndParamFileInfo getCommandLine(SpawnAction linkOrArchiveAction) {
    // Commandlines are a pair of a SingletonCommandLine with tool path and
    // a CommandLine with rest of command line
    // The latter optionally specifies a ParamFile
    return linkOrArchiveAction.getCommandLines().unpack().get(1);
  }
}
