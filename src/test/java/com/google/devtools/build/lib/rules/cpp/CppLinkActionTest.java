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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
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
          public Artifact getDerivedArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
            return CppLinkActionTest.this.getDerivedArtifact(
                rootRelativePath, root, ActionsTestUtil.NULL_ARTIFACT_OWNER);
          }
        },
        masterConfig);
  }

  private final FeatureConfiguration getMockFeatureConfiguration(RuleContext ruleContext)
      throws Exception {
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
            .build();

    ImmutableList<CToolchain.ActionConfig> actionConfigs =
        CppActionConfigs.getLegacyActionConfigs(
            CppPlatform.LINUX,
            "gcc_tool",
            "ar_tool",
            "strip_tool",
            /* supportsInterfaceSharedLibraries= */ false);

    return CcToolchainFeaturesTest.buildFeatures(ruleContext, features, actionConfigs)
        .getFeatureConfiguration(
            ImmutableSet.of(
                Link.LinkTargetType.EXECUTABLE.getActionName(),
                Link.LinkTargetType.NODEPS_DYNAMIC_LIBRARY.getActionName(),
                Link.LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
                Link.LinkTargetType.STATIC_LIBRARY.getActionName()));
  }

  @Test
  public void testToolchainFeatureFlags() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                ruleContext,
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
        CcToolchainFeaturesTest.buildFeatures(
                ruleContext,
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
        "  srcs = ['some-dir/bar.so', 'some-other-dir/qux.so'],",
        "  linkopts = [",
        "    '-ldl',",
        "    '-lutil',",
        "  ],",
        ")");
    scratch.file("x/some-dir/bar.so");
    scratch.file("x/some-other-dir/qux.so");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:foo");
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/foo");

    List<String> arguments = linkAction.getLinkCommandLine().arguments();

    assertThat(Joiner.on(" ").join(arguments))
        .matches(
            ".* -L[^ ]*some-dir(?= ).* -L[^ ]*some-other-dir(?= ).* "
                + "-lbar -lqux(?= ).* -ldl -lutil .*");
    assertThat(Joiner.on(" ").join(arguments))
        .matches(".* -Wl,-rpath[^ ]*some-dir(?= ).* -Wl,-rpath[^ ]*some-other-dir .*");
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
    CppLinkAction linkAction =
        (CppLinkAction)
            getGeneratingAction(configuredTarget, "x/foo");

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
  public void testCompilesTestSourcesIntoDynamicLibrary() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Skip the test on Windows.
      // TODO(bazel-team): maybe we should move that test that doesn't work with MSVC toolchain to
      // its own suite with a TestSpec?
      return;
    }
    scratch.file(
        "x/BUILD",
        "cc_test(name = 'a', srcs = ['a.cc'])",
        "cc_binary(name = 'b', srcs = ['a.cc'], linkstatic = 0)");
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--experimental_link_compile_output_separately", "--force_pic");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    String cpu = CrosstoolConfigurationHelper.defaultCpu();
    CppLinkAction linkAction =
        (CppLinkAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.ifso");
    assertThat(linkAction.getArguments())
        .contains(
            getBinArtifactWithNoOwner("_solib_" + cpu + "/libx_Sliba.ifso").getExecPathString());
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.so");

    configuredTarget = getConfiguredTarget("//x:b");
    linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/b");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/b/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/b");
  }

  @Test
  public void testCompilesDynamicModeTestSourcesWithoutFeatureIntoDynamicLibrary()
      throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Skip the test on Windows.
      // TODO(bazel-team): maybe we should move that test that doesn't work with MSVC toolchain to
      // its own suite with a TestSpec?
      return;
    }
    scratch.file(
        "x/BUILD",
        "cc_test(name = 'a', srcs = ['a.cc'], features = ['-static_link_test_srcs'])",
        "cc_binary(name = 'b', srcs = ['a.cc'])");
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--force_pic");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    String cpu = CrosstoolConfigurationHelper.defaultCpu();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.ifso");
    assertThat(linkAction.getArguments())
        .contains(
            getBinArtifactWithNoOwner("_solib_" + cpu + "/libx_Sliba.ifso").getExecPathString());
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.so");

    configuredTarget = getConfiguredTarget("//x:b");
    linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/b");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/b/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/b");
  }

  @Test
  public void testCompilesDynamicModeBinarySourcesWithoutFeatureIntoDynamicLibrary()
      throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Skip the test on Windows.
      // TODO(bazel-team): maybe we should move that test that doesn't work with MSVC toolchain to
      // its own suite with a TestSpec?
      return;
    }
    scratch.file(
        "x/BUILD", "cc_binary(name = 'a', srcs = ['a.cc'], features = ['-static_link_test_srcs'])");
    scratch.file("x/a.cc", "int main() {}");
    useConfiguration("--force_pic", "--dynamic_mode=default");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//x:a");
    String cpu = CrosstoolConfigurationHelper.defaultCpu();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/a");
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .doesNotContain("bin _solib_" + cpu + "/libx_Sliba.ifso");
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/a/a.pic.o");
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/a");
  }

  @Test
  public void testToolchainFeatureEnv() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                ruleContext,
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
    FAKE,
    RUNTIME_SOLIB_DIR
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
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration(ruleContext);

    ActionTester.runTest(
        NonStaticAttributes.class,
        new ActionCombinationFactory<NonStaticAttributes>() {

          @Override
          public Action generate(ImmutableSet<NonStaticAttributes> attributesToFlip)
              throws InterruptedException {
            CcToolchainProvider toolchain =
                CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    attributesToFlip.contains(NonStaticAttributes.OUTPUT_FILE)
                        ? dynamicOutputFile
                        : staticOutputFile,
                    toolchain,
                    toolchain.getFdoProvider(),
                    featureConfiguration,
                    MockCppSemantics.INSTANCE) {};
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
            builder.setFake(attributesToFlip.contains(NonStaticAttributes.FAKE));
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
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration(ruleContext);

    ActionTester.runTest(
        StaticKeyAttributes.class,
        new ActionCombinationFactory<StaticKeyAttributes>() {

          @Override
          public Action generate(ImmutableSet<StaticKeyAttributes> attributes)
              throws InterruptedException {
            CcToolchainProvider toolchain =
                CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                        ? staticOutputFile
                        : dynamicOutputFile,
                    toolchain,
                    toolchain.getFdoProvider(),
                    featureConfiguration,
                    MockCppSemantics.INSTANCE) {};
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
    Artifact output = getDerivedArtifact(
        PathFragment.create("output/path.xyz"), getTargetConfiguration().getBinDirectory(
            RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
    final Artifact outputIfso = getDerivedArtifact(
        PathFragment.create("output/path.ifso"), getTargetConfiguration().getBinDirectory(
            RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
            ruleContext,
            output,
            toolchain,
            toolchain.getFdoProvider(),
            FeatureConfiguration.EMPTY,
            MockCppSemantics.INSTANCE);
    builder.setLinkType(LinkTargetType.STATIC_LIBRARY);
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
   * Links a small target.
   * Checks that resource estimates are above the minimum and scale correctly.
   */
  @Test
  public void testSmallLocalLinkResourceEstimate() throws Exception {
    assertLinkSizeAccuracy(3);
  }

  /**
   * Fake links a large target.
   * Checks that resource estimates are above the minimum and scale correctly.
   * The actual link action is irrelevant; we are just checking the estimate.
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
                getMockFeatureConfiguration(ruleContext))
            .setFake(true)
            .build();

    // Ensure that minima are enforced.
    ResourceSet resources = linkAction.estimateResourceConsumptionLocal();
    assertThat(resources.getMemoryMb())
        .isAtLeast(CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb());
    assertThat(resources.getCpuUsage())
        .isAtLeast(CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage());

    final int linkSize = Iterables.size(linkAction.getLinkCommandLine().getLinkerInputArtifacts());
    ResourceSet scaledSet = ResourceSet.createWithRamCpu(
        CppLinkAction.LINK_RESOURCES_PER_INPUT.getMemoryMb() * linkSize,
        CppLinkAction.LINK_RESOURCES_PER_INPUT.getCpuUsage() * linkSize
    );

    // Ensure that anything above the minimum is properly scaled.
    assertThat(resources.getMemoryMb() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb()
        || resources.getMemoryMb() == scaledSet.getMemoryMb()).isTrue();
    assertThat(resources.getCpuUsage() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage()
        || resources.getCpuUsage() == scaledSet.getCpuUsage()).isTrue();
  }

  private CppLinkActionBuilder createLinkBuilder(
      RuleContext ruleContext,
      Link.LinkTargetType type,
      String outputPath,
      Iterable<Artifact> nonLibraryInputs,
      ImmutableList<LibraryToLink> libraryInputs,
      FeatureConfiguration featureConfiguration)
      throws Exception {
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
                ruleContext,
                new Artifact(
                    PathFragment.create(outputPath),
                    getTargetConfiguration()
                        .getBinDirectory(ruleContext.getRule().getRepository())),
                ruleContext.getConfiguration(),
                toolchain,
                toolchain.getFdoProvider(),
                featureConfiguration,
                MockCppSemantics.INSTANCE)
            .addObjectFiles(nonLibraryInputs)
            .addLibraries(NestedSetBuilder.wrap(Order.LINK_ORDER, libraryInputs))
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
        getMockFeatureConfiguration(ruleContext));
  }

  public Artifact getOutputArtifact(String relpath) {
    return new Artifact(
        getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
        getTargetConfiguration().getBinFragment().getRelative(relpath));
  }

  private Artifact scratchArtifact(String s) {
    Path execRoot = outputBase.getRelative("exec");
    Path outputRoot = execRoot.getRelative("out");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, outputRoot);
    try {
      return new Artifact(scratch.overwriteFile(outputRoot.getRelative(s).toString()), root);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void assertError(String expectedSubstring, CppLinkActionBuilder builder)
      throws InterruptedException {
    try {
      builder.build();
      fail();
    } catch (RuntimeException e) {
      assertThat(e).hasMessageThat().contains(expectedSubstring);
    }
  }

  @Test
  public void testInterfaceOutputWithoutBuildingDynamicLibraryIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.EXECUTABLE)
            .setInterfaceOutput(scratchArtifact("FakeInterfaceOutput"));

    assertError("Interface output can only be used with non-fake DYNAMIC_LIBRARY targets", builder);
  }

  @Test
  public void testInterfaceOutputForDynamicLibrary() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                ruleContext,
                "supports_interface_shared_objects: true ",
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
  public void testStaticLinkWithSymbolsCountOutputIsError() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLinkingMode(LinkingMode.STATIC)
            .setLibraryIdentifier("foo")
            .setSymbolCountsOutput(scratchArtifact("dummySymbolCounts"));

    assertError("the symbol counts output must be null for static links", builder);
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
    return new SpecialArtifact(
        ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative("out")),
        execPath,
        ArtifactOwner.NullArtifactOwner.INSTANCE,
        SpecialArtifactType.TREE);
  }

  private void verifyArguments(
      Iterable<String> arguments,
      Iterable<String> allowedArguments,
      Iterable<String> disallowedArguments) {
    assertThat(arguments).containsAllIn(allowedArguments);
    assertThat(arguments).containsNoneIn(disallowedArguments);
  }

  @Test
  public void testLinksTreeArtifactLibraries() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    SpecialArtifact testTreeArtifact = createTreeArtifact("library_directory");

    TreeFileArtifact library0 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library1.o");

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
  public void testLinksTreeArtifactLibraryForDeps() throws Exception {
    // This test only makes sense if start/end lib archives are supported.
    analysisMock.ccSupport().setupCrosstool(mockToolsConfig, "supports_start_end_lib: true");
    useConfiguration("--start_end_lib");
    RuleContext ruleContext = createDummyRuleContext();

    SpecialArtifact testTreeArtifact = createTreeArtifact("library_directory");

    TreeFileArtifact library0 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library1.o");

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

    Artifact archiveFile = scratchArtifact("library.a");

    CppLinkActionBuilder builder =
        createLinkBuilder(ruleContext, LinkTargetType.STATIC_LIBRARY)
            .setLibraryIdentifier("foo")
            .addLibrary(
                LinkerInputs.newInputLibrary(
                    archiveFile,
                    ArtifactCategory.STATIC_LIBRARY,
                    null,
                    ImmutableList.<Artifact>of(testTreeArtifact),
                    new LtoCompilationContext(ImmutableMap.of()),
                    null,
                    /* mustKeepDebug= */ false));

    CppLinkAction linkAction = builder.build();

    Iterable<String> treeArtifactsPaths = ImmutableList.of(testTreeArtifact.getExecPathString());
    Iterable<String> treeFileArtifactsPaths =
        ImmutableList.of(library0.getExecPathString(), library1.getExecPathString());

    // Should only reference the tree artifact.
    verifyArguments(
        linkAction.getLinkCommandLine().getRawLinkArgv(),
        treeArtifactsPaths,
        treeFileArtifactsPaths);
    verifyArguments(linkAction.getArguments(), treeArtifactsPaths, treeFileArtifactsPaths);

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

    TreeFileArtifact library0 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library0.o");
    TreeFileArtifact library1 = ActionInputHelper.treeFileArtifact(testTreeArtifact, "library1.o");

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
      Artifact output = scratchArtifact("output." + linkType.getDefaultExtension());

      CppLinkActionBuilder builder =
          createLinkBuilder(
                  ruleContext,
                  linkType,
                  output.getExecPathString(),
                  ImmutableList.<Artifact>of(),
                  ImmutableList.<LibraryToLink>of(),
                  getMockFeatureConfiguration(ruleContext))
              .setLibraryIdentifier("foo")
              .addObjectFiles(ImmutableList.of(testTreeArtifact))
              .addObjectFile(objectFile)
              // Makes sure this doesn't use a params file.
              .setFake(true);

      CppLinkAction linkAction = builder.build();
      assertThat(linkAction.getCommandLine(expander))
          .containsAllOf(
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
                getMockFeatureConfiguration(ruleContext))
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
                getMockFeatureConfiguration(ruleContext))
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

    String solibPrefix = "_solib_" + CrosstoolConfigurationHelper.defaultCpu();
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
                getMockFeatureConfiguration(ruleContext))
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

    assertThat(linkAction.getLinkCommandLine().getLinkopts()).isEmpty();
  }

  @Test
  public void testSplitExecutableLinkCommand() throws Exception {
    RuleContext ruleContext = createDummyRuleContext();

    CppLinkAction linkAction = createLinkBuilder(ruleContext, LinkTargetType.EXECUTABLE).build();
    Pair<List<String>, List<String>> result = linkAction.getLinkCommandLine().splitCommandline();

    String linkCommandLine = Joiner.on(" ").join(result.first);
    assertThat(linkCommandLine).contains("gcc_tool");
    assertThat(linkCommandLine).contains("-o");
    assertThat(linkCommandLine).contains("output/path.a");
    assertThat(linkCommandLine).contains("path.a-2.params");

    assertThat(result.second).contains("-lstdc++");
  }
}
