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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionConfigs.CppLinkPlatform;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.Staticness;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CppLinkAction}. */
@RunWith(JUnit4.class)
public class CppLinkActionTest extends BuildViewTestCase {
  private RuleContext createDummyRuleContext() throws Exception {
    return view.getRuleContextForTesting(reporter, scratchConfiguredTarget(
        "dummyRuleContext", "dummyRuleContext",
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
        }, masterConfig);
  }
  
  private final FeatureConfiguration getMockFeatureConfiguration() throws Exception {
    return CcToolchainFeaturesTest.buildFeatures(
            CppLinkActionConfigs.getCppLinkActionConfigs(CppLinkPlatform.LINUX))
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
                "out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration,
                false)
            .build();
    assertThat(linkAction.getArgv()).contains("some_flag");
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
                "out",
                ImmutableList.<Artifact>of(),
                ImmutableList.<LibraryToLink>of(),
                featureConfiguration,
                false)
            .build();
    assertThat(linkAction.getEnvironment()).containsEntry("foo", "bar");
  }
  
  /**
   * This mainly checks that non-static links don't have identical keys. Many options are only
   * allowed on non-static links, and we test several of them here.
   */
  @Test
  public void testComputeKeyNonStatic() throws Exception {
    final RuleContext ruleContext = createDummyRuleContext();
    final PathFragment exeOutputPath = new PathFragment("dummyRuleContext/output/path");
    final PathFragment dynamicOutputPath = new PathFragment("dummyRuleContext/output/path.so");
    final Artifact staticOutputFile = getBinArtifactWithNoOwner(exeOutputPath.getPathString());
    final Artifact dynamicOutputFile = getBinArtifactWithNoOwner(dynamicOutputPath.getPathString());
    final Artifact oFile = getSourceArtifact("cc/a.o");
    final Artifact oFile2 = getSourceArtifact("cc/a2.o");
    final Artifact interfaceSoBuilder = getBinArtifactWithNoOwner("foo/build_interface_so");
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration();

    ActionTester.runTest(
        128,
        new ActionCombinationFactory() {

          @Override
          public Action generate(int i) {
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(ruleContext, (i & 2) == 0
                    ? dynamicOutputFile : staticOutputFile) {
                  @Override
                  protected Artifact getInterfaceSoBuilder() {
                    return interfaceSoBuilder;
                  }
                };
            builder.addCompilationInputs(
                (i & 1) == 0 ? ImmutableList.of(oFile) : ImmutableList.of(oFile2));
            if ((i & 2) == 0) {
              builder.setLinkType(LinkTargetType.DYNAMIC_LIBRARY);
              builder.setLibraryIdentifier("foo");
            } else {
              builder.setLinkType(LinkTargetType.EXECUTABLE);
            }
            builder.setLinkStaticness(LinkStaticness.DYNAMIC);
            builder.setNativeDeps((i & 4) == 0);
            builder.setUseTestOnlyFlags((i & 8) == 0);
            builder.setWholeArchive((i & 16) == 0);
            builder.setFake((i & 32) == 0);
            builder.setRuntimeSolibDir((i & 64) == 0 ? null : new PathFragment("so1"));
            builder.setFeatureConfiguration(featureConfiguration);

            return builder.build();
          }
        });
  }

  /**
   * This mainly checks that static library links don't have identical keys, and it also compares
   * them with simple dynamic library links.
   */
  @Test
  public void testComputeKeyStatic() throws Exception {
    final RuleContext ruleContext = createDummyRuleContext();
    final PathFragment staticOutputPath = new PathFragment("dummyRuleContext/output/path.a");
    final PathFragment dynamicOutputPath = new PathFragment("dummyRuleContext/output/path.so");
    final Artifact staticOutputFile = getBinArtifactWithNoOwner(staticOutputPath.getPathString());
    final Artifact dynamicOutputFile = getBinArtifactWithNoOwner(dynamicOutputPath.getPathString());
    final Artifact oFile = getSourceArtifact("cc/a.o");
    final Artifact oFile2 = getSourceArtifact("cc/a2.o");
    final Artifact interfaceSoBuilder = getBinArtifactWithNoOwner("foo/build_interface_so");
    final FeatureConfiguration featureConfiguration = getMockFeatureConfiguration();

    ActionTester.runTest(
        4,
        new ActionCombinationFactory() {

          @Override
          public Action generate(int i) {
            CppLinkActionBuilder builder =
                new CppLinkActionBuilder(ruleContext, (i & 2) == 0
                    ? staticOutputFile : dynamicOutputFile) {
                  @Override
                  protected Artifact getInterfaceSoBuilder() {
                    return interfaceSoBuilder;
                  }
                };
            builder.addCompilationInputs(
                (i & 1) == 0 ? ImmutableList.of(oFile) : ImmutableList.of(oFile2));
            builder.setLinkType(
                (i & 2) == 0 ? LinkTargetType.STATIC_LIBRARY : LinkTargetType.DYNAMIC_LIBRARY);
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
        new PathFragment("output/path.xyz"), getTargetConfiguration().getBinDirectory(
            RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
    final Artifact outputIfso = getDerivedArtifact(
        new PathFragment("output/path.ifso"), getTargetConfiguration().getBinDirectory(
            RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
    CppLinkActionBuilder builder = new CppLinkActionBuilder(ruleContext, output);
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
                "binary2",
                objects.build(),
                ImmutableList.<LibraryToLink>of(),
                new FeatureConfiguration(),
                false)
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
      FeatureConfiguration featureConfiguration,
      boolean shouldIncludeToolchain)
      throws Exception {
    RuleContext ruleContext = createDummyRuleContext();
    CppLinkActionBuilder builder =
        new CppLinkActionBuilder(
                ruleContext,
                new Artifact(
                    new PathFragment(outputPath),
                    getTargetConfiguration().getBinDirectory(
                        ruleContext.getRule().getRepository())),
                ruleContext.getConfiguration(),
                shouldIncludeToolchain ? CppHelper.getToolchain(ruleContext) : null)
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
    PathFragment output = new PathFragment("dummyRuleContext/output/path.a");
    return createLinkBuilder(
        type,
        output.getPathString(),
        ImmutableList.<Artifact>of(),
        ImmutableList.<LibraryToLink>of(),
        new FeatureConfiguration(),
        true);
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

  private void assertError(String expectedSubstring, CppLinkActionBuilder builder) {
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
