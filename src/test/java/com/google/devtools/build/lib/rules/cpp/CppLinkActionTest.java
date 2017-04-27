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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
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
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
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
          public Artifact getDerivedArtifact(PathFragment rootRelativePath, Root root) {
            return CppLinkActionTest.this.getDerivedArtifact(
                rootRelativePath, root, ActionsTestUtil.NULL_ARTIFACT_OWNER);
          }
        },
        masterConfig);
  }

  private final FeatureConfiguration getMockFeatureConfiguration() throws Exception {
    return CcToolchainFeaturesTest.buildFeatures(
            CppActionConfigs.getCppActionConfigs(
                CppPlatform.LINUX,
                ImmutableSet.<String>of(),
                "gcc_tool",
                "dynamic_library_linker_tool",
                true))
        .getFeatureConfiguration(
            Link.LinkTargetType.EXECUTABLE.getActionName(),
            Link.LinkTargetType.DYNAMIC_LIBRARY.getActionName(),
            Link.LinkTargetType.STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.PIC_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY.getActionName(),
            Link.LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY.getActionName());
  }

  @Test
  public void testToolchainFeatureFlags() throws Exception {
    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                "feature {",
                "   name: 'a'",
                "   flag_set {",
                "      action: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "      flag_group { flag: 'some_flag' }",
                "   }",
                "}",
                "action_config {",
                "   config_name: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "   action_name: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "   tool {",
                "      tool_path: 'toolchain/mock_tool'",
                "   }",
                "}")
            .getFeatureConfiguration("a", Link.LinkTargetType.EXECUTABLE.getActionName());

    CppLinkAction linkAction =
        createLinkBuilder(
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .build();
    assertThat(linkAction.getArgv()).contains("some_flag");
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
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/foo"
        + OsUtils.executableExtension());

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
            getGeneratingAction(configuredTarget, "x/foo" + OsUtils.executableExtension());

    Iterable<? extends VariableValue> runtimeLibrarySearchDirectories =
        linkAction
            .getLinkCommandLine()
            .getBuildVariables()
            .getSequenceVariable(CppLinkActionBuilder.RUNTIME_LIBRARY_SEARCH_DIRECTORIES_VARIABLE);
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
    String extension = OsUtils.executableExtension();
    CppLinkAction linkAction =
        (CppLinkAction) getGeneratingAction(configuredTarget, "x/a" + extension);
    assertThat(artifactsToStrings(linkAction.getInputs()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.ifso");
    assertThat(linkAction.getArguments())
        .contains(
            getBinArtifactWithNoOwner("_solib_" + cpu + "/libx_Sliba.ifso").getExecPathString());
    RunfilesProvider runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .contains("bin _solib_" + cpu + "/libx_Sliba.so");

    configuredTarget = getConfiguredTarget("//x:b");
    linkAction = (CppLinkAction) getGeneratingAction(configuredTarget, "x/b" + extension);
    assertThat(artifactsToStrings(linkAction.getInputs())).contains("bin x/_objs/b/x/a.pic.o");
    runfilesProvider = configuredTarget.getProvider(RunfilesProvider.class);
    assertThat(artifactsToStrings(runfilesProvider.getDefaultRunfiles().getArtifacts()))
        .containsExactly("bin x/b");
  }

  @Test
  public void testToolchainFeatureEnv() throws Exception {
    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                "feature {",
                "   name: 'a'",
                "   env_set {",
                "      action: '" + Link.LinkTargetType.EXECUTABLE.getActionName() + "'",
                "      env_entry { key: 'foo', value: 'bar' }",
                "   }",
                "}")
            .getFeatureConfiguration("a");

    CppLinkAction linkAction =
        createLinkBuilder(
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .build();
    assertThat(linkAction.getEnvironment()).containsEntry("foo", "bar");
  }

  private enum NonStaticAttributes {
    OUTPUT_FILE,
    COMPILATION_INPUTS,
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
    final Artifact oFile = getSourceArtifact("cc/a.o");
    final Artifact oFile2 = getSourceArtifact("cc/a2.o");
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration();

    ActionTester.runTest(
        NonStaticAttributes.class,
        new ActionCombinationFactory<NonStaticAttributes>() {

          @Override
          public Action generate(ImmutableSet<NonStaticAttributes> attributesToFlip)
              throws InterruptedException {
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    attributesToFlip.contains(NonStaticAttributes.OUTPUT_FILE)
                        ? dynamicOutputFile
                        : staticOutputFile,
                    CppHelper.getToolchain(ruleContext, ":cc_toolchain"),
                    CppHelper.getFdoSupport(ruleContext, ":cc_toolchain")) {};
            builder.addCompilationInputs(
                attributesToFlip.contains(NonStaticAttributes.COMPILATION_INPUTS)
                    ? ImmutableList.of(oFile)
                    : ImmutableList.of(oFile2));
            if (attributesToFlip.contains(NonStaticAttributes.OUTPUT_FILE)) {
              builder.setLinkType(LinkTargetType.DYNAMIC_LIBRARY);
              builder.setLibraryIdentifier("foo");
            } else {
              builder.setLinkType(LinkTargetType.EXECUTABLE);
            }
            builder.setLinkStaticness(LinkStaticness.DYNAMIC);
            builder.setNativeDeps(attributesToFlip.contains(NonStaticAttributes.NATIVE_DEPS));
            builder.setUseTestOnlyFlags(
                attributesToFlip.contains(NonStaticAttributes.USE_TEST_ONLY_FLAGS));
            builder.setFake(attributesToFlip.contains(NonStaticAttributes.FAKE));
            builder.setRuntimeSolibDir(
                attributesToFlip.contains(NonStaticAttributes.RUNTIME_SOLIB_DIR)
                    ? null
                    : PathFragment.create("so1"));
            builder.setFeatureConfiguration(featureConfiguration);

            return builder.build();
          }
        });
  }

  private enum StaticKeyAttributes {
    OUTPUT_FILE,
    COMPILATION_INPUTS
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
    final Artifact oFile = getSourceArtifact("cc/a.o");
    final Artifact oFile2 = getSourceArtifact("cc/a2.o");
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration();

    ActionTester.runTest(
        StaticKeyAttributes.class,
        new ActionCombinationFactory<StaticKeyAttributes>() {

          @Override
          public Action generate(ImmutableSet<StaticKeyAttributes> attributes)
              throws InterruptedException {
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(
                    ruleContext,
                    attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                        ? staticOutputFile
                        : dynamicOutputFile,
                    CppHelper.getToolchain(ruleContext, ":cc_toolchain"),
                    CppHelper.getFdoSupport(ruleContext, ":cc_toolchain")) {};
            builder.addCompilationInputs(
                attributes.contains(StaticKeyAttributes.COMPILATION_INPUTS)
                    ? ImmutableList.of(oFile)
                    : ImmutableList.of(oFile2));
            builder.setLinkType(
                attributes.contains(StaticKeyAttributes.OUTPUT_FILE)
                    ? LinkTargetType.STATIC_LIBRARY
                    : LinkTargetType.DYNAMIC_LIBRARY);
            builder.setLibraryIdentifier("foo");
            builder.setFeatureConfiguration(featureConfiguration);
            return builder.build();
          }
        });
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
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
            ruleContext,
            output,
            CppHelper.getToolchain(ruleContext, ":cc_toolchain"),
            CppHelper.getFdoSupport(ruleContext, ":cc_toolchain"));
    builder.setLinkType(LinkTargetType.STATIC_LIBRARY);
    assertTrue(builder.canSplitCommandLine());

    builder.setLinkType(LinkTargetType.DYNAMIC_LIBRARY);
    assertTrue(builder.canSplitCommandLine());

    builder.setInterfaceOutput(outputIfso);
    assertFalse(builder.canSplitCommandLine());

    builder.setInterfaceOutput(null);
    builder.setLinkType(LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
    assertFalse(builder.canSplitCommandLine());
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
    ImmutableList.Builder<Artifact> objects = ImmutableList.builder();
    for (int i = 0; i < inputs; i++) {
      objects.add(getOutputArtifact("object" + i + ".o"));
    }

    CppLinkAction linkAction =
        createLinkBuilder(
                Link.LinkTargetType.EXECUTABLE,
                "dummyRuleContext/binary2",
                objects.build(),
                ImmutableList.<LibraryToLink>of(),
                new FeatureConfiguration())
            .setFake(true)
            .build();

    // Ensure that minima are enforced.
    ResourceSet resources = linkAction.estimateResourceConsumptionLocal();
    assertTrue(resources.getMemoryMb() >= CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb());
    assertTrue(resources.getCpuUsage() >= CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage());
    assertTrue(resources.getIoUsage() >= CppLinkAction.MIN_STATIC_LINK_RESOURCES.getIoUsage());

    final int linkSize = Iterables.size(linkAction.getLinkCommandLine().getLinkerInputs());
    ResourceSet scaledSet = ResourceSet.createWithRamCpuIo(
        CppLinkAction.LINK_RESOURCES_PER_INPUT.getMemoryMb() * linkSize,
        CppLinkAction.LINK_RESOURCES_PER_INPUT.getCpuUsage() * linkSize,
        CppLinkAction.LINK_RESOURCES_PER_INPUT.getIoUsage() * linkSize
    );

    // Ensure that anything above the minimum is properly scaled.
    assertTrue(resources.getMemoryMb() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getMemoryMb()
      || resources.getMemoryMb() == scaledSet.getMemoryMb());
    assertTrue(resources.getCpuUsage() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getCpuUsage()
      || resources.getCpuUsage() == scaledSet.getCpuUsage());
    assertTrue(resources.getIoUsage() == CppLinkAction.MIN_STATIC_LINK_RESOURCES.getIoUsage()
      || resources.getIoUsage() == scaledSet.getIoUsage());
  }

  private CppLinkActionBuilder createLinkBuilder(
      Link.LinkTargetType type,
      String outputPath,
      Iterable<Artifact> nonLibraryInputs,
      ImmutableList<LibraryToLink> libraryInputs,
      FeatureConfiguration featureConfiguration)
      throws Exception {
    RuleContext ruleContext = createDummyRuleContext();
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
                ruleContext,
                new Artifact(
                    PathFragment.create(outputPath),
                    getTargetConfiguration()
                        .getBinDirectory(ruleContext.getRule().getRepository())),
                ruleContext.getConfiguration(),
                CppHelper.getToolchain(ruleContext, ":cc_toolchain"),
                CppHelper.getFdoSupport(ruleContext, ":cc_toolchain"))
            .addObjectFiles(nonLibraryInputs)
            .addLibraries(NestedSetBuilder.wrap(Order.LINK_ORDER, libraryInputs))
            .setLinkType(type)
            .setCrosstoolInputs(NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER))
            .setLinkStaticness(
                type.staticness() == Staticness.STATIC
                    ? LinkStaticness.FULLY_STATIC
                    : LinkStaticness.MOSTLY_STATIC)
            .setFeatureConfiguration(featureConfiguration);
    return builder;
  }

  private CppLinkActionBuilder createLinkBuilder(Link.LinkTargetType type) throws Exception {
    PathFragment output = PathFragment.create("dummyRuleContext/output/path.a");
    return createLinkBuilder(
        type,
        output.getPathString(),
        ImmutableList.<Artifact>of(),
        ImmutableList.<LibraryToLink>of(),
        new FeatureConfiguration());
  }

  public Artifact getOutputArtifact(String relpath) {
    return new Artifact(
        getTargetConfiguration().getBinDirectory(RepositoryName.MAIN).getPath()
            .getRelative(relpath),
        getTargetConfiguration().getBinDirectory(RepositoryName.MAIN),
        getTargetConfiguration().getBinFragment().getRelative(relpath));
  }

  private Artifact scratchArtifact(String s) {
    try {
      return new Artifact(
          scratch.overwriteFile(outputBase.getRelative("WORKSPACE").getRelative(s).toString()),
          Root.asDerivedRoot(scratch.dir(outputBase.getRelative("WORKSPACE").toString())));
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
      assertThat(e.getMessage()).contains(expectedSubstring);
    }
  }

  @Test
  public void testInterfaceOutputWithoutBuildingDynamicLibraryIsError() throws Exception {
    CppLinkActionBuilder builder =
        createLinkBuilder(LinkTargetType.EXECUTABLE)
            .setInterfaceOutput(scratchArtifact("FakeInterfaceOutput"));

    assertError("Interface output can only be used with non-fake DYNAMIC_LIBRARY targets", builder);
  }

  @Test
  public void testInterfaceOutputForDynamicLibrary() throws Exception {
    FeatureConfiguration featureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                "feature {",
                "   name: 'build_interface_libraries'",
                "   flag_set {",
                "       action: '" + LinkTargetType.DYNAMIC_LIBRARY.getActionName() + "',",
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
                "       action: 'c++-link-dynamic-library'",
                "       flag_group {",
                "           flag: 'dynamic_library_linker_tool'",
                "       }",
                "   }",
                "}",
                "feature {",
                "    name: 'has_configured_linker_path'",
                "}",
                "action_config {",
                "   config_name: '" + LinkTargetType.DYNAMIC_LIBRARY.getActionName() + "'",
                "   action_name: '" + LinkTargetType.DYNAMIC_LIBRARY.getActionName() + "'",
                "   tool {",
                "       tool_path: 'custom/crosstool/scripts/link_dynamic_library.sh'",
                "   }",
                "   implies: 'has_configured_linker_path'",
                "   implies: 'build_interface_libraries'",
                "   implies: 'dynamic_library_linker_tool'",
                "}")
            .getFeatureConfiguration(
                "build_interface_libraries",
                "dynamic_library_linker_tool",
                LinkTargetType.DYNAMIC_LIBRARY.getActionName());
    CppLinkActionBuilder builder =
        createLinkBuilder(
                LinkTargetType.DYNAMIC_LIBRARY,
                "foo.so",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration)
            .setLibraryIdentifier("foo")
            .setInterfaceOutput(scratchArtifact("FakeInterfaceOutput.ifso"));

    List<String> commandLine = builder.build().getCommandLine();
    assertThat(commandLine).hasSize(6);
    assertThat(commandLine.get(0)).endsWith("custom/crosstool/scripts/link_dynamic_library.sh");
    assertThat(commandLine.get(1)).isEqualTo("yes");
    assertThat(commandLine.get(2)).endsWith("tools/cpp/build_interface_so");
    assertThat(commandLine.get(3)).endsWith("foo.so");
    assertThat(commandLine.get(4)).isEqualTo("FakeInterfaceOutput.ifso");
    assertThat(commandLine.get(5)).isEqualTo("dynamic_library_linker_tool");
  }

  @Test
  public void testStaticLinkWithDynamicIsError() throws Exception {
    CppLinkActionBuilder builder =
        createLinkBuilder(LinkTargetType.STATIC_LIBRARY)
            .setLinkStaticness(LinkStaticness.DYNAMIC)
            .setLibraryIdentifier("foo");

    assertError("static library link must be static", builder);
  }

  @Test
  public void testStaticLinkWithSymbolsCountOutputIsError() throws Exception {
    CppLinkActionBuilder builder =
        createLinkBuilder(LinkTargetType.STATIC_LIBRARY)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier("foo")
            .setSymbolCountsOutput(scratchArtifact("dummySymbolCounts"));

    assertError("the symbol counts output must be null for static links", builder);
  }

  @Test
  public void testStaticLinkWithNativeDepsIsError() throws Exception {
    CppLinkActionBuilder builder =
        createLinkBuilder(LinkTargetType.STATIC_LIBRARY)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier("foo")
            .setNativeDeps(true);

    assertError("the native deps flag must be false for static links", builder);
  }

  @Test
  public void testStaticLinkWithWholeArchiveIsError() throws Exception {
    CppLinkActionBuilder builder =
        createLinkBuilder(LinkTargetType.STATIC_LIBRARY)
            .setLinkStaticness(LinkStaticness.FULLY_STATIC)
            .setLibraryIdentifier("foo")
            .setWholeArchive(true);

    assertError("the need whole archive flag must be false for static links", builder);
  }
}
