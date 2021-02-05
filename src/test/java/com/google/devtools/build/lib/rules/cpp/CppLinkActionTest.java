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
import com.google.common.primitives.Ints;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.EnvEntry;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CppLinkAction}. */
@RunWith(JUnit4.class)
public class CppLinkActionTest extends BuildViewTestCase {

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
          public void registerAction(ActionAnalysisMetadata... action) {
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
        },
        masterConfig);
  }

  private FeatureConfiguration getMockFeatureConfiguration(ImmutableMap<String, String> envVars) {
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
    for (String envVar : envVars.keySet()) {
      envSet.addEnvEntry(
          EnvEntry.newBuilder().setKey(envVar).setValue(envVars.get(envVar)).build());
    }

    CToolchain.Feature linkCppStandardLibrary =
        CToolchain.Feature.newBuilder()
            .setName("link_cpp_standard_library")
            .setEnabled(true)
            .addFlagSet(flagSet)
            .addEnvSet(envSet.build())
            .build();
    ImmutableList<CToolchain.Feature> features =
        new ImmutableList.Builder<CToolchain.Feature>()
            .addAll(
                CppActionConfigs.getLegacyFeatures(
                    CppPlatform.LINUX,
                    ImmutableSet.of(),
                    "dynamic_library_linker_tool",
                    /* supportsEmbeddedRuntimes= */ true,
                    /* supportsInterfaceSharedLibraries= */ false,
                    /* doNotSplitLinkingCmdline= */ true))
            .addAll(
                CppActionConfigs.getFeaturesToAppearLastInFeaturesList(
                    ImmutableSet.of(), /* doNotSplitLinkingCmdline= */ true))
            .add(linkCppStandardLibrary)
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
                  LinkTargetType.EXECUTABLE.getActionName(),
                  LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName(),
                  LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
                  LinkTargetType.STATIC_LIBRARY.getActionName()));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testToolchainFeatureFlags() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainTestHelper.buildFeatures(
                MockCcSupport.EMPTY_EXECUTABLE_ACTION_CONFIG,
                "feature {",
                "   name: 'a'",
                "   flag_set {",
                "      action: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "      flag_group { flag: 'some_flag' }",
                "   }",
                "}")
            .getFeatureConfiguration(
                ImmutableSet.of("a", Link.LinkTargetType.EXECUTABLE.getActionName()));

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .build();
    assertThat(linkAction.getArguments()).contains("some_flag");
  }

  @Test
  public void testExecutionRequirementsFromCrosstool() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainTestHelper.buildFeatures(
                "action_config {",
                "   config_name: '" + LinkTargetType.EXECUTABLE.getActionName() + "'",
                "   action_name: '" + LinkTargetType.EXECUTABLE.getActionName() + "'",
                "   tool {",
                "      tool_path: 'DUMMY_TOOL'",
                "      execution_requirement: 'dummy-exec-requirement'",
                "   }",
                "}")
            .getFeatureConfiguration(ImmutableSet.of(LinkTargetType.EXECUTABLE.getActionName()));

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.of(),
                ImmutableList.of(),
                featureConfiguration)
            .build();
    assertThat(linkAction.getExecutionInfo()).containsEntry("dummy-exec-requirement", "");
  }

  @Test
  public void testLibOptsAndLibSrcsAreInCorrectOrder() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'foo',",
        "  srcs = ['some-dir/libbar.so', 'some-other-dir/qux.so'],",
        "  linkopts = [",
        "    '-ldl',",
        "    '-lutil',",
        "  ],",
        ")");
    scratch.file("x/some-dir/libbar.so");
    scratch.file("x/some-other-dir/qux.so");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:foo");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/foo");

    List<String> arguments = linkAction.getLinkCommandLine().arguments();

    assertThat(Joiner.on(" ").join(arguments))
        .matches(
            ".* -L[^ ]*some-dir(?= ).* -L[^ ]*some-other-dir(?= ).* "
                + "-lbar -l:qux.so(?= ).* -ldl -lutil .*");
    assertThat(Joiner.on(" ").join(arguments))
        .matches(".* -Wl,-rpath[^ ]*some-dir(?= ).* -Wl,-rpath[^ ]*some-other-dir .*");
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
        "cc_binary(",
        "  name = 'libfoo.so',",
        "  srcs = ['foo.cc'],",
        "  linkshared = 1,",
        "  linkstatic = 0,",
        ")");
    useConfiguration("--legacy_whole_archive");
    assertThat(getLibfooArguments()).doesNotContain("-Wl,-whole-archive");
  }

  private List<String> getLibfooArguments() throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:libfoo.so");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/libfoo.so");
    return linkAction.getArguments();
  }

  @Test
  public void testExposesRuntimeLibrarySearchDirectoriesVariable() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'foo',",
        "  srcs = ['some-dir/bar.so', 'some-other-dir/qux.so'],",
        ")");
    scratch.file("x/some-dir/bar.so");
    scratch.file("x/some-other-dir/qux.so");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:foo");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/foo");

    Iterable<? extends VariableValue> runtimeLibrarySearchDirectories =
        linkAction
            .getLinkCommandLine()
            .getBuildVariables()
            .getSequenceVariable(
                LinkBuildVariables.RUNTIME_LIBRARY_SEARCH_DIRECTORIES.getVariableName());
    List<String> directories = new ArrayList<>();
    for (VariableValue value : runtimeLibrarySearchDirectories) {
      directories.add(value.getStringValue("runtime_library_search_directory"));
    }
    assertThat(Joiner.on(" ").join(directories)).matches(".*some-dir .*some-other-dir");
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
        "cc_test(name='a', srcs=['a.cc'], features=['dynamic_link_test_srcs'])",
        "cc_binary(name='b', srcs=['a.cc'])",
        "cc_test(name='c', srcs=['a.cc'], features=['dynamic_link_test_srcs'], linkstatic=1)");
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--force_pic");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin _solib_k8/libx_Sliba.ifso");
    assertThat(linkAction.getArguments())
        .contains(getBinArtifactWithNoOwner("_solib_k8/libx_Sliba.ifso").getExecPathString());
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .contains("bin _solib_k8/libx_Sliba.so");

    configuredTarget = getConfiguredTarget("//x:b");
    linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/b");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/b/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/b");

    configuredTarget = getConfiguredTarget("//x:c");
    linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/c");
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
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .doesNotContain("bin _solib_k8/libx_Sliba.ifso");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/a/a.pic.o");
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/a");
  }

  @Test
  public void testToolchainFeatureEnv() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainTestHelper.buildFeatures(
                MockCcSupport.EMPTY_EXECUTABLE_ACTION_CONFIG,
                "feature {",
                "   name: 'a'",
                "   env_set {",
                "      action: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "      env_entry { key: 'foo', value: 'bar' }",
                "   }",
                "}")
            .getFeatureConfiguration(
                ImmutableSet.of(Link.LinkTargetType.EXECUTABLE.getActionName(), "a"));

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .build();
    assertThat(linkAction.getIncompleteEnvironmentForTesting()).containsEntry("foo", "bar");
  }

  private enum NonStaticAttributes {
    OUTPUT_FILE,
    NATIVE_DEPS,
    USE_TEST_ONLY_FLAGS,
    RUNTIME_SOLIB_DIR,
    ENVIRONMENT,
  }

  /**
   * This mainly checks that non-static links don't have identical keys. Many options are only
   * allowed on non-static links, and we test several of them here.
   */
  @Test
  public void testComputeKeyNonStatic() throws Exception {
    final RuleContext ruleContext = createDummyRuleContext();
    final PathFragment exeOutputPath = PathFragment.create("dummyRuleContext/output/path");
    final PathFragment dynamicOutputPath = PathFragment.create("dummyRuleContext/output/path.so");
    final Artifact staticOutputFile = getBinArtifactWithNoOwner(exeOutputPath.getPathString());
    final Artifact dynamicOutputFile = getBinArtifactWithNoOwner(dynamicOutputPath.getPathString());

    ActionTester.runTest(
        NonStaticAttributes.class,
        new ActionCombinationFactory<NonStaticAttributes>() {

          @Override
          public Action generate(ImmutableSet<NonStaticAttributes> attributesToFlip)
              throws InterruptedException, RuleErrorException {
            CcToolchainProvider toolchain =
                CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    ruleContext,
                    ruleContext.getLabel(),
                    attributesToFlip.contains(NonStaticAttributes.OUTPUT_FILE)
                        ? dynamicOutputFile
                        : staticOutputFile,
                    ruleContext.getConfiguration(),
                    toolchain,
                    toolchain.getFdoContext(),
                    getMockFeatureConfiguration(
                        attributesToFlip.contains(NonStaticAttributes.ENVIRONMENT)
                            ? ImmutableMap.of("var", "value")
                            : ImmutableMap.of()),
                    MockCppSemantics.INSTANCE);
            if (attributesToFlip.contains(NonStaticAttributes.OUTPUT_FILE)) {
              builder.setLinkType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY);
              builder.setLibraryIdentifier("foo");
            } else {
              builder.setLinkType(LinkTargetType.EXECUTABLE);
            }
            builder.setLinkingMode(Link.LinkingMode.DYNAMIC);
            builder.setNativeDeps(attributesToFlip.contains(NonStaticAttributes.NATIVE_DEPS));
            builder.setUseTestOnlyFlags(
                attributesToFlip.contains(NonStaticAttributes.USE_TEST_ONLY_FLAGS));
            builder.setToolchainLibrariesSolibDir(
                attributesToFlip.contains(NonStaticAttributes.RUNTIME_SOLIB_DIR)
                    ? null
                    : PathFragment.create("so1"));

            return builder.build();
          }
        },
        actionKeyContext);
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
              throws InterruptedException, RuleErrorException {
            CcToolchainProvider toolchain =
                CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    ruleContext,
                    ruleContext.getLabel(),
                    attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                        ? staticOutputFile
                        : dynamicOutputFile,
                    ruleContext.getConfiguration(),
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
          }
        },
        actionKeyContext);
  }

  @Test
  public void testCommandLineSplitting() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();
    Artifact output =
        getDerivedArtifact(
            PathFragment.create("output/path.xyz"),
            getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
            ActionsTestUtil.NULL_ARTIFACT_OWNER);
    final Artifact outputIfso =
        getDerivedArtifact(
            PathFragment.create("output/path.ifso"),
            getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
            ActionsTestUtil.NULL_ARTIFACT_OWNER);
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
            ruleContext,
            ruleContext,
            ruleContext.getLabel(),
            output,
            ruleContext.getConfiguration(),
            toolchain,
            toolchain.getFdoContext(),
            FeatureConfiguration.EMPTY,
            MockCppSemantics.INSTANCE);
    builder.setLinkType(LinkTargetType.STATIC_LIBRARY);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setLinkType(LinkTargetType.OBJC_ARCHIVE);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setLinkType(LinkTargetType.OBJC_EXECUTABLE);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setLinkType(LinkTargetType.OBJCPP_EXECUTABLE);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setLinkType(LinkTargetType.OBJC_FULLY_LINKED_ARCHIVE);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setLinkType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY);
    assertThat(builder.canSplitCommandLine()).isTrue();

    builder.setInterfaceOutput(outputIfso);
    assertThat(builder.canSplitCommandLine()).isFalse();

    builder.setInterfaceOutput(null);
    builder.setLinkType(LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
    assertThat(builder.canSplitCommandLine()).isFalse();
  }

  /**
   * Links a small target. Checks that resource estimates are above the minimum and scale correctly.
   */
  @Test
  public void testSmallLocalLinkResourceEstimate() throws Exception {
    assertLinkSizeAccuracy(3);
  }

  /**
   * Fake links a large target. Checks that resource estimates are above the minimum and scale
   * correctly. The actual link action is irrelevant; we are just checking the estimate.
   */
  @Test
  public void testLargeLocalLinkResourceEstimate() throws Exception {
    assertLinkSizeAccuracy(7000);
  }

  private void assertLinkSizeAccuracy(int inputs) throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    ImmutableList.Builder<Artifact> objects = ImmutableList.builder();
    for (int i = 0; i < inputs; i++) {
      objects.add(getOutputArtifact("object" + i + ".o"));
    }

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/binary2",
                objects.build(),
                ImmutableList.<LibraryToLink>of(),
                getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()))
            .build();

    // Ensure that minima are enforced.
    ResourceSet resources = linkAction.estimateResourceConsumptionLocal();
    assertThat(resources.getMemoryMb())
        .isAtLeast(CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb());
    assertThat(resources.getCpuUsage())
        .isAtLeast(CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage());

    final int linkSize =
        linkAction.getLinkCommandLine().getLinkerInputArtifacts().memoizedFlattenAndGetSize();
    ResourceSet scaledSet =
        ResourceSet.createWithRamCpu(
            CppLinkAction.LINK_RESOURCES_PER_INPUT.getMemoryMb() * linkSize,
            CppLinkAction.LINK_RESOURCES_PER_INPUT.getCpuUsage() * linkSize);

    // Ensure that anything above the minimum is properly scaled.
    assertThat(
            resources.getMemoryMb() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb()
                || resources.getMemoryMb() == scaledSet.getMemoryMb())
        .isTrue();
    assertThat(
            resources.getCpuUsage() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage()
                || resources.getCpuUsage() == scaledSet.getCpuUsage())
        .isTrue();
  }

  private CppLinkActionBuilder createLinkBuilder(
      RuleContext ruleContext,
      Link.LinkTargetType type,
      String outputPath,
      Iterable<Artifact> nonLibraryInputs,
      ImmutableList<LibraryToLink> libraryInputs,
      FeatureConfiguration featureConfiguration) {
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
                ruleContext,
                ruleContext,
                ruleContext.getLabel(),
                ActionsTestUtil.createArtifact(
                    getTargetConfiguration().getBinDirectory(ruleContext.getRule().getRepository()),
                    outputPath),
                ruleContext.getConfiguration(),
                toolchain,
                toolchain.getFdoContext(),
                featureConfiguration,
                MockCppSemantics.INSTANCE)
            .addObjectFiles(nonLibraryInputs)
            .addLibraries(libraryInputs)
            .setLinkType(type)
            .setLinkerFiles(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
            .setLinkingMode(LinkingMode.STATIC);
    return builder;
  }

  private CppLinkActionBuilder createLinkBuilder(RuleContext ruleContext, Link.LinkTargetType type)
      throws Exception {
    PathFragment output = PathFragment.create("dummyRuleContext/output/path.a");
    return createLinkBuilder(
        ruleContext,
        type,
        output.getPathString(),
        ImmutableList.<Artifact>of(),
        ImmutableList.<LibraryToLink>of(),
        getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()));
  }

  public Artifact getOutputArtifact(String relpath) {
    return ActionsTestUtil.createArtifactWithExecPath(
        getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
        getTargetConfiguration().getBinFragment(RepositoryName.MAIN).getRelative(relpath));
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
    Exception e = assertThrows(Exception.class, () -> builder.build());
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
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainTestHelper.buildFeatures(
                MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE,
                "feature {",
                "   name: 'build_interface_libraries'",
                "   flag_set {",
                "       action: '" + LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName() + "',",
                "       flag_group {",
                "           flag: '%{generate_interface_library}'",
                "           flag: '%{interface_library_builder_path}'",
                "           flag: '%{interface_library_input_path}'",
                "           flag: '%{interface_library_output_path}'",
                "       }",
                "   }",
                "}",
                "feature {",
                "   name: 'dynamic_library_linker_tool'",
                "   flag_set {",
                "       action: 'c++-link-nodeps-dynamic-library'",
                "       flag_group {",
                "           flag: 'dynamic_library_linker_tool'",
                "       }",
                "   }",
                "}",
                "feature {",
                "    name: 'has_configured_linker_path'",
                "}",
                "action_config {",
                "   config_name: '" + LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName() + "'",
                "   action_name: '" + LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName() + "'",
                "   tool {",
                "       tool_path: 'custom/crosstool/scripts/link_dynamic_library.sh'",
                "   }",
                "   implies: 'has_configured_linker_path'",
                "   implies: 'build_interface_libraries'",
                "   implies: 'dynamic_library_linker_tool'",
                "}")
            .getFeatureConfiguration(
                ImmutableSet.of(
                    "build_interface_libraries",
                    "dynamic_library_linker_tool",
                    LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName()));
    CppLinkActionBuilder builder =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.NODEPS_DYNAMIC_LIBRARY,
                "foo.so",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .setLibraryIdentifier("foo")
            .setInterfaceOutput(scratchArtifact("FakeInterfaceOutput.ifso"));

    List<String> commandLine = builder.build().getCommandLine(null);
    assertThat(commandLine).hasSize(6);
    assertThat(commandLine.get(0)).endsWith("custom/crosstool/scripts/link_dynamic_library.sh");
    assertThat(commandLine.get(1)).isEqualTo("yes");
    assertThat(commandLine.get(2)).endsWith("tools/cpp/build_interface_so");
    assertThat(commandLine.get(3)).endsWith("foo.so");
    assertThat(commandLine.get(4)).isEqualTo("out/FakeInterfaceOutput.ifso");
    assertThat(commandLine.get(5)).isEqualTo("dynamic_library_linker_tool");
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

  private SpecialArtifact createTreeArtifact(String name) {
    FileSystem fs = scratch.getFileSystem();
    Path execRoot = fs.getPath(TestUtils.tmpDir());
    PathFragment execPath = PathFragment.create("out").getRelative(name);
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out"), execPath);
  }

  private void verifyArguments(
      Iterable<String> arguments,
      Iterable<String> allowedArguments,
      Iterable<String> disallowedArguments) {
    assertThat(arguments).containsAtLeastElementsIn(allowedArguments);
    assertThat(arguments).containsNoneIn(disallowedArguments);
  }

  @Test
  public void testLinksTreeArtifactLibraries() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

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

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLibraryIdentifier("foo")
            .addObjectFiles(ImmutableList.of(testTreeArtifact));

    CppLinkAction linkAction = builder.build();

    Iterable<String> treeArtifactsPaths = ImmutableList.of(testTreeArtifact.getExecPathString());
    Iterable<String> treeFileArtifactsPaths =
        ImmutableList.of(library0.getExecPathString(), library1.getExecPathString());

    // Should only reference the tree artifact.
    verifyArguments(
        linkAction.getLinkCommandLine().getRawLinkArgv(),
        treeArtifactsPaths,
        treeFileArtifactsPaths);

    // Should only reference tree file artifacts.
    verifyArguments(
        linkAction.getLinkCommandLine().getRawLinkArgv(expander),
        treeFileArtifactsPaths,
        treeArtifactsPaths);
  }

  @Test
  public void testStaticLinking() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    ImmutableList<LinkTargetType> targetTypesToTest =
        ImmutableList.of(
            LinkTargetType.STATIC_LIBRARY,
            LinkTargetType.PIC_STATIC_LIBRARY,
            LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY,
            LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY);

    SpecialArtifact testTreeArtifact = createTreeArtifact("library_directory");

    TreeFileArtifact library0 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = TreeFileArtifact.createTreeOutput(testTreeArtifact, "library1.o");

    ArtifactExpander expander =
        (artifact, output) -> {
          if (artifact.equals(testTreeArtifact)) {
            output.add(library0);
            output.add(library1);
          }
        };

    Artifact objectFile = scratchArtifact("objectFile.o");

    for (LinkTargetType linkType : targetTypesToTest) {

      scratch.deleteFile("dummyRuleContext/BUILD");
      Artifact output = ruleContext.getBinArtifact("output." + linkType.getDefaultExtension());

      CppLinkActionBuilder builder =
          createLinkBuilder(
                  ruleContext,
                  linkType,
                  output.getRootRelativePathString(),
                  ImmutableList.<Artifact>of(),
                  ImmutableList.<LibraryToLink>of(),
                  getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()))
              .setLibraryIdentifier("foo")
              .addObjectFiles(ImmutableList.of(testTreeArtifact))
              .addObjectFile(objectFile);

      CppLinkAction linkAction = builder.build();
      assertThat(
              ImmutableList.copyOf(
                  linkAction.getLinkCommandLine().paramCmdLine().arguments(expander)))
          .containsAtLeast(
              library0.getExecPathString(),
              library1.getExecPathString(),
              objectFile.getExecPathString())
          .inOrder();
    }
  }

  /** Tests that -pie is removed when -shared is also present (http://b/5611891#). */
  @Test
  public void testPieOptionDisabledForSharedLibraries() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.DYNAMIC_LIBRARY,
                "dummyRuleContext/out.so",
                ImmutableList.of(),
                ImmutableList.of(),
                getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()))
            .setLinkingMode(Link.LinkingMode.STATIC)
            .addLinkopts(ImmutableList.of("-pie", "-other", "-pie"))
            .setLibraryIdentifier("foo")
            .build();

    List<String> argv = linkAction.getLinkCommandLine().getRawLinkArgv();
    assertThat(argv).doesNotContain("-pie");
    assertThat(argv).contains("-other");
  }

  /** Tests that -pie is removed when -shared is also present (http://b/5611891#). */
  @Test
  public void testPieOptionKeptForExecutables() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.of(),
                ImmutableList.of(),
                getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()))
            .setLinkingMode(Link.LinkingMode.STATIC)
            .addLinkopts(ImmutableList.of("-pie", "-other", "-pie"))
            .build();

    List<String> argv = linkAction.getLinkCommandLine().getRawLinkArgv();
    assertThat(argv).contains("-pie");
    assertThat(argv).contains("-other");
  }

  @Test
  public void testLinkoptsComeAfterLinkerInputs() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    String solibPrefix = "_solib_k8";
    Iterable<LibraryToLink> linkerInputs =
        LinkerInputs.opaqueLibrariesToLink(
            ArtifactCategory.DYNAMIC_LIBRARY,
            ImmutableList.of(
                getOutputArtifact(solibPrefix + "/FakeLinkerInput1.so"),
                getOutputArtifact(solibPrefix + "/FakeLinkerInput2.so"),
                getOutputArtifact(solibPrefix + "/FakeLinkerInput3.so"),
                getOutputArtifact(solibPrefix + "/FakeLinkerInput4.so")));

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.of(),
                ImmutableList.copyOf(linkerInputs),
                getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of()))
            .addLinkopts(ImmutableList.of("FakeLinkopt1", "FakeLinkopt2"))
            .build();

    List<String> argv = linkAction.getLinkCommandLine().getRawLinkArgv();
    int lastLinkerInputIndex =
        Ints.max(
            argv.indexOf("FakeLinkerInput1"), argv.indexOf("FakeLinkerInput2"),
            argv.indexOf("FakeLinkerInput3"), argv.indexOf("FakeLinkerInput4"));
    int firstLinkoptIndex = Math.min(argv.indexOf("FakeLinkopt1"), argv.indexOf("FakeLinkopt2"));
    assertThat(lastLinkerInputIndex).isLessThan(firstLinkoptIndex);
  }

  @Test
  public void testLinkoptsAreOmittedForStaticLibrary() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkAction linkAction =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .addLinkopt("FakeLinkopt1")
            .setLibraryIdentifier("foo")
            .build();

    assertThat(MockCcSupport.getLinkopts(linkAction.getLinkCommandLine())).isEmpty();
  }

  @Test
  public void testSplitExecutableLinkCommandStatic() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.DO_NOT_SPLIT_LINKING_CMDLINE));

    RuleContext ruleContext = createDummyRuleContext();

    CppLinkAction linkAction = createLinkBuilder(ruleContext, LinkTargetType.EXECUTABLE).build();
    Pair<List<String>, List<String>> result = linkAction.getLinkCommandLine().splitCommandline();

    String linkCommandLine = Joiner.on(" ").join(result.first);
    assertThat(linkCommandLine).contains("gcc_tool");
    assertThat(linkCommandLine).contains("-o");
    assertThat(linkCommandLine).contains("output/path.a");
    assertThat(linkCommandLine).contains("path.a-2.params");

    assertThat(result.second).contains("-lcpp_standard_library");
  }

  private String removeOutDirectory(String s) {
    return s.replace("blaze-out", "").replace("bazel-out", "");
  }

  @Test
  public void testSplitExecutableLinkCommandDynamicWithNoSplitting() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.DO_NOT_SPLIT_LINKING_CMDLINE));
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        getMockFeatureConfiguration(/* envVars= */ ImmutableMap.of());

    CppLinkAction linkAction =
        createLinkBuilder(
                ruleContext,
                LinkTargetType.DYNAMIC_LIBRARY,
                "dummyRuleContext/out.so",
                ImmutableList.of(),
                ImmutableList.of(),
                featureConfiguration)
            .setLibraryIdentifier("library")
            .build();
    Pair<List<String>, List<String>> result = linkAction.getLinkCommandLine().splitCommandline();

    assertThat(
            result.first.stream()
                .map(x -> removeOutDirectory(x))
                .collect(ImmutableList.toImmutableList()))
        .containsExactly(
            "crosstool/gcc_tool", "@/k8-fastbuild/bin/dummyRuleContext/out.so-2.params")
        .inOrder();
    assertThat(
            result.second.stream()
                .map(x -> removeOutDirectory(x))
                .collect(ImmutableList.toImmutableList()))
        .containsExactly(
            "-shared",
            "-o",
            "/k8-fastbuild/bin/dummyRuleContext/out.so",
            "-Wl,-S",
            "--sysroot=/usr/grte/v1")
        .inOrder();
  }

  @Test
  public void testExposesLinkstampSources() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'bin',",
        "  deps = [':lib'],",
        ")",
        "cc_library(",
        "  name = 'lib',",
        "  linkstamp = 'linkstamp.cc',",
        ")");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:bin");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/bin");
    assertThat(artifactsToStrings(linkAction.getLinkstampObjects()))
        .containsExactly("src x/linkstamp.cc");
  }

  @Test
  public void testExposesLinkstampObjects() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "  name = 'bin',",
        "  deps = [':lib'],",
        ")",
        "cc_library(",
        "  name = 'lib',",
        "  linkstamp = 'linkstamp.cc',",
        ")");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:bin");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/bin");
    assertThat(artifactsToStrings(linkAction.getLinkstampObjectFileInputs()))
        .containsExactly("bin x/_objs/bin/x/linkstamp.o");
  }
}
