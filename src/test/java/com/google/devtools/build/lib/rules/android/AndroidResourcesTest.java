// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AndroidResourcesTest} */
@RunWith(JUnit4.class)
public class AndroidResourcesTest extends ResourceTestBase {
  private static final PathFragment DEFAULT_RESOURCE_ROOT = PathFragment.create(RESOURCE_ROOT);
  private static final ImmutableList<PathFragment> RESOURCES_ROOTS =
      ImmutableList.of(DEFAULT_RESOURCE_ROOT);

  private static final ImmutableSet<String> TOOL_FILENAMES = ImmutableSet.of("aapt2", "empty.sh");

  @Before
  @Test
  public void testGetResourceRootsNoResources() throws Exception {
    assertThat(getResourceRoots()).isEmpty();
  }

  @Test
  public void testGetResourceRootsInvalidResourceDirectory() throws Exception {
    try {
      getResourceRoots("is-this-drawable-or-values/foo.xml");
      assertWithMessage("Expected exception not thrown!").fail();
    } catch (RuleErrorException e) {
      // expected
    }

    errorConsumer.assertAttributeError(
        "resource_files", "is not in the expected resource directory structure");
  }

  @Test
  public void testGetResourceRootsMultipleRoots() throws Exception {
    try {
      getResourceRoots("subdir/values/foo.xml", "otherdir/values/bar.xml");
      assertWithMessage("Expected exception not thrown!").fail();
    } catch (RuleErrorException e) {
      // expected
    }

    errorConsumer.assertAttributeError(
        "resource_files", "All resources must share a common directory");
  }

  @Test
  public void testGetResourceRoots() throws Exception {
    assertThat(getResourceRoots("values-hdpi/foo.xml", "values-mdpi/bar.xml"))
        .isEqualTo(RESOURCES_ROOTS);
  }

  @Test
  public void testGetResourceRootsCommonSubdirectory() throws Exception {
    assertThat(getResourceRoots("subdir/values-hdpi/foo.xml", "subdir/values-mdpi/bar.xml"))
        .containsExactly(DEFAULT_RESOURCE_ROOT.getRelative("subdir"));
  }

  private ImmutableList<PathFragment> getResourceRoots(String... pathResourceStrings)
      throws Exception {
    return getResourceRoots(getResources(pathResourceStrings));
  }

  private ImmutableList<PathFragment> getResourceRoots(ImmutableList<Artifact> artifacts)
      throws Exception {
    return AndroidResources.getResourceRoots(errorConsumer, artifacts, "resource_files");
  }

  @Test
  public void testFilterEmpty() throws Exception {
    assertFilter(ImmutableList.of(), ImmutableList.of());
  }

  @Test
  public void testFilterNoop() throws Exception {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "values-es/bar.xml");
    assertFilter(resources, resources);
  }

  @Test
  public void testFilterToEmpty() throws Exception {
    assertFilter(getResources("values-en/foo.xml", "values-es/bar.xml"), ImmutableList.of());
  }

  @Test
  public void testPartiallyFilter() throws Exception {
    Artifact keptResource = getResource("values-en/foo.xml");
    assertFilter(
        ImmutableList.of(keptResource, getResource("values-es/bar.xml")),
        ImmutableList.of(keptResource));
  }

  @Test
  public void testFilterIsDependency() throws Exception {
    Artifact keptResource = getResource("values-en/foo.xml");
    assertFilter(
        ImmutableList.of(keptResource, getResource("drawable/bar.png")),
        ImmutableList.of(keptResource),
        /* isDependency = */ true);
  }

  private void assertFilter(
      ImmutableList<Artifact> unfilteredResources, ImmutableList<Artifact> filteredResources)
      throws Exception {
    assertFilter(unfilteredResources, filteredResources, /* isDependency = */ false);
  }

  private void assertFilter(
      ImmutableList<Artifact> unfilteredResources,
      ImmutableList<Artifact> filteredResources,
      boolean isDependency)
      throws Exception {
    ImmutableList<PathFragment> unfilteredResourcesRoots = getResourceRoots(unfilteredResources);
    AndroidResources unfiltered =
        new AndroidResources(unfilteredResources, unfilteredResourcesRoots);

    ImmutableList.Builder<Artifact> filteredDepsBuilder = ImmutableList.builder();

    ResourceFilter fakeFilter =
        ResourceFilter.of(ImmutableSet.copyOf(filteredResources), filteredDepsBuilder::add);

    Optional<AndroidResources> filtered = unfiltered.maybeFilter(fakeFilter, isDependency);

    if (filteredResources.equals(unfilteredResources)) {
      // We expect filtering to have been a no-op
      assertThat(filtered.isPresent()).isFalse();
    } else {
      // The resources and their roots should be filtered
      assertThat(filtered.get().getResources())
          .containsExactlyElementsIn(filteredResources)
          .inOrder();
      assertThat(filtered.get().getResourceRoots())
          .containsExactlyElementsIn(getResourceRoots(filteredResources))
          .inOrder();
    }

    if (!isDependency) {
      // The filter should not record any filtered dependencies
      assertThat(filteredDepsBuilder.build()).isEmpty();
    } else {
      // The filtered dependencies should be exactly the list of filtered resources
      assertThat(unfilteredResources)
          .containsExactlyElementsIn(
              Iterables.concat(filteredDepsBuilder.build(), filteredResources));
    }
  }

  @Test
  public void testParseNoCompile() throws Exception {
    useConfiguration("--android_aapt=aapt");

    RuleContext ruleContext = getRuleContext(/* useDataBinding = */ true);
    ParsedAndroidResources parsed = assertParse(ruleContext);

    // Since we are not using aapt2, there should be no compiled symbols
    assertThat(parsed.getCompiledSymbols()).isNull();

    // The parse action should take resources in and output symbols
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ parsed.getResources(),
        /* outputs = */ ImmutableList.of(parsed.getSymbols()));
  }

  @Test
  public void testParseAndCompile() throws Exception {
    mockAndroidSdkWithAapt2();
    useConfiguration("--android_sdk=//sdk:sdk", "--android_aapt=aapt2");

    RuleContext ruleContext = getRuleContext(/* useDataBinding = */ false);
    ParsedAndroidResources parsed = assertParse(ruleContext);

    assertThat(parsed.getCompiledSymbols()).isNotNull();

    // The parse action should take resources in and output symbols
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ parsed.getResources(),
        /* outputs = */ ImmutableList.of(parsed.getSymbols()));

    // Since there was no data binding, the compile action should just take in resources and output
    // compiled symbols.
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ parsed.getResources(),
        /* outputs = */ ImmutableList.of(parsed.getCompiledSymbols()));
  }

  @Test
  public void testParseWithDataBinding() throws Exception {
    mockAndroidSdkWithAapt2();
    useConfiguration("--android_sdk=//sdk:sdk", "--android_aapt=aapt2");

    RuleContext ruleContext = getRuleContext(/* useDataBinding = */ true);

    ParsedAndroidResources parsed = assertParse(ruleContext);

    // The parse action should take resources and busybox artifacts in and output symbols
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ parsed.getResources(),
        /* outputs = */ ImmutableList.of(parsed.getSymbols()));

    // The compile action should take in resources and manifest in and output compiled symbols and
    // an unused data binding zip.
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.<Artifact>builder()
            .addAll(parsed.getResources())
            .add(getManifest())
            .build(),
        /* outputs = */ ImmutableList.of(
            parsed.getCompiledSymbols(),
            DataBinding.getSuffixedInfoFile(ruleContext, "_unused")));
  }

  /**
   * Assets that the action used to generate the given outputs has the expected inputs and outputs.
   */
  private void assertActionArtifacts(
      RuleContext ruleContext, ImmutableList<Artifact> inputs, ImmutableList<Artifact> outputs) {
    // Actions must have at least one output
    assertThat(outputs).isNotEmpty();

    // Get the action from one of the outputs
    ActionAnalysisMetadata action =
        ruleContext.getAnalysisEnvironment().getLocalGeneratingAction(outputs.get(0));
    assertThat(action).isNotNull();

    assertThat(removeToolingArtifacts(action.getInputs())).containsExactlyElementsIn(inputs);

    assertThat(action.getOutputs()).containsExactlyElementsIn(outputs);
  }

  /** Remove busybox and aapt2 tooling artifacts from a list of action inputs */
  private Iterable<Artifact> removeToolingArtifacts(Iterable<Artifact> inputArtifacts) {
    return Streams.stream(inputArtifacts)
        .filter(
            artifact ->
                // Not a known tool
                !TOOL_FILENAMES.contains(artifact.getFilename())
                    // Not one of the various busybox tools (we get different ones on different OSs)
                    && !artifact.getFilename().contains("busybox")
                    // Not a params file
                    && !artifact.getFilename().endsWith(".params"))
        .collect(Collectors.toList());
  }

  /**
   * Validates that a parse action was invoked correctly. Returns the {@link ParsedAndroidResources}
   * for further validation.
   */
  private ParsedAndroidResources assertParse(RuleContext ruleContext) throws Exception {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "drawable-hdpi/bar.png");
    AndroidResources raw =
        new AndroidResources(
            resources, AndroidResources.getResourceRoots(ruleContext, resources, "resource_files"));
    StampedAndroidManifest manifest =
        new StampedAndroidManifest(ruleContext, getManifest(), "some.java.pkg", false);

    ParsedAndroidResources parsed = raw.parse(ruleContext, manifest);

    // Inherited values should be equal
    assertThat(raw).isEqualTo(parsed);

    // Label should be set from RuleContext
    assertThat(parsed.getLabel()).isEqualTo(ruleContext.getLabel());

    return parsed;
  }

  private Artifact getManifest() {
    return getResource("some/path/AndroidManifest.xml");
  }

  /** Gets a dummy rule context object by creating a dummy target. */
  private RuleContext getRuleContext(boolean useDataBinding) throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java/foo",
            "target",
            "android_library(name = 'target',",
            useDataBinding ? "  enable_data_binding = True" : "",
            ")");
    RuleContext dummy = getRuleContext(target);

    ExtendedEventHandler eventHandler = new StoredEventHandler();
    assertThat(targetConfig.isActionsEnabled()).isTrue();
    return view.getRuleContextForTesting(
        eventHandler,
        target,
        new CachingAnalysisEnvironment(
            view.getArtifactFactory(),
            skyframeExecutor.getActionKeyContext(),
            ConfiguredTargetKey.of(target.getLabel(), targetConfig),
            /*isSystemEnv=*/ false,
            targetConfig.extendedSanityChecks(),
            eventHandler,
            /*env=*/ null,
            targetConfig.isActionsEnabled()),
        new BuildConfigurationCollection(
            ImmutableList.of(dummy.getConfiguration()), dummy.getHostConfiguration()));
  }
}
