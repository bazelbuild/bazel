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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidResourcesTest.WithPlatforms;
import com.google.devtools.build.lib.rules.android.AndroidResourcesTest.WithoutPlatforms;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Tests {@link AndroidResources} */
@RunWith(Suite.class)
@SuiteClasses({WithoutPlatforms.class, WithPlatforms.class})
public abstract class AndroidResourcesTest extends ResourceTestBase {
  /** Use legacy toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithoutPlatforms extends AndroidResourcesTest {}

  /** Use platform-based toolchain resolution. */
  @RunWith(JUnit4.class)
  public static class WithPlatforms extends AndroidResourcesTest {
    @Override
    protected boolean platformBasedToolchains() {
      return true;
    }
  }

  private static final PathFragment DEFAULT_RESOURCE_ROOT = PathFragment.create(RESOURCE_ROOT);
  private static final ImmutableList<PathFragment> RESOURCES_ROOTS =
      ImmutableList.of(DEFAULT_RESOURCE_ROOT);

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

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

  @Test
  public void testFilterValidatedNoop() throws Exception {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "values-es/bar.xml");
    assertFilterValidated(resources, resources);
  }

  @Test
  public void testFilterValidated() throws Exception {
    Artifact keptResource = getResource("values-en/foo.xml");
    assertFilterValidated(
        ImmutableList.of(keptResource, getResource("drawable/bar.png")),
        ImmutableList.of(keptResource));
  }

  private void assertFilterValidated(
      ImmutableList<Artifact> unfilteredResources, ImmutableList<Artifact> filteredResources)
      throws Exception {
    RuleContext ruleContext = getRuleContext();
    final AndroidDataContext dataContext = AndroidDataContext.forNative(ruleContext);
    ValidatedAndroidResources unfiltered =
        new AndroidResources(unfilteredResources, getResourceRoots(unfilteredResources))
            .process(
                ruleContext,
                dataContext,
                getManifest(),
                DataBinding.contextFrom(ruleContext, dataContext.getAndroidConfig()),
                /* neverlink = */ false);
    Optional<? extends AndroidResources> maybeFiltered =
        assertFilter(unfiltered, filteredResources, /* isDependency = */ true);

    if (maybeFiltered.isPresent()) {
      AndroidResources filtered = maybeFiltered.get();
      assertThat(filtered instanceof ValidatedAndroidResources).isTrue();
      ValidatedAndroidResources validated = (ValidatedAndroidResources) filtered;

      // Validate fields related to validation are unchanged
      assertThat(validated.getRTxt()).isEqualTo(unfiltered.getRTxt());
      assertThat(validated.getAapt2RTxt()).isEqualTo(unfiltered.getAapt2RTxt());
    }
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
    AndroidResources unfiltered =
        new AndroidResources(unfilteredResources, getResourceRoots(unfilteredResources));
    assertFilter(unfiltered, filteredResources, isDependency);
  }

  private Optional<? extends AndroidResources> assertFilter(
      AndroidResources unfiltered, ImmutableList<Artifact> filteredResources, boolean isDependency)
      throws Exception {

    ImmutableList.Builder<Artifact> filteredDepsBuilder = ImmutableList.builder();

    ResourceFilter fakeFilter =
        ResourceFilter.of(ImmutableSet.copyOf(filteredResources), filteredDepsBuilder::add);

    Optional<? extends AndroidResources> filtered =
        unfiltered.maybeFilter(errorConsumer, fakeFilter, isDependency);

    if (filteredResources.equals(unfiltered.getResources())) {
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
      assertThat(unfiltered.getResources())
          .containsExactlyElementsIn(
              Iterables.concat(filteredDepsBuilder.build(), filteredResources));
    }

    return filtered;
  }

  @Test
  public void testParseAndCompile() throws Exception {
    RuleContext ruleContext = getRuleContext();
    ParsedAndroidResources parsed = assertParse(ruleContext);

    assertThat(parsed.getCompiledSymbols()).isNotNull();

    // Since there was no data binding, the compile action should just take in resources and output
    // compiled symbols.
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ parsed.getResources(),
        /* outputs = */ ImmutableList.of(parsed.getCompiledSymbols()));
  }

  @Test
  public void testParseWithDataBinding() throws Exception {
    RuleContext ruleContext = getRuleContextWithDataBinding();

    ParsedAndroidResources parsed = assertParse(ruleContext);

    // The compile action should take in resources and manifest in and output compiled symbols and
    // a databinding zip.
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.<Artifact>builder()
            .addAll(parsed.getResources())
            .add(parsed.getManifest())
            .build(),
        /* outputs = */ ImmutableList.of(
            parsed.getCompiledSymbols(), DataBinding.getLayoutInfoFile(ruleContext)));
  }

  @Test
  public void testMergeCompiled() throws Exception {
    RuleContext ruleContext = getRuleContext();
    ParsedAndroidResources parsed = assertParse(ruleContext);
    MergedAndroidResources merged =
        parsed.merge(
            AndroidDataContext.forNative(ruleContext),
            ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false));

    // Besides processed manifest, inherited values should be equal
    assertThat(parsed).isEqualTo(new ParsedAndroidResources(merged, parsed.getStampedManifest()));

    // There should be a new processed manifest
    assertThat(merged.getManifest()).isNotEqualTo(parsed.getManifest());

    assertThat(merged.getDataBindingInfoZip()).isNull();
    assertThat(merged.getCompiledSymbols()).isNotNull();

    // We use the compiled symbols file to build the resource class jar
    assertActionArtifacts(
        ruleContext,
        /*inputs=*/ ImmutableList.of(merged.getCompiledSymbols(), parsed.getManifest()),
        /*outputs=*/ ImmutableList.of(merged.getClassJar(), merged.getManifest()));
  }

  @Test
  public void testValidateAapt2() throws Exception {
    RuleContext ruleContext = getRuleContext();

    MergedAndroidResources merged = makeMergedResources(ruleContext);
    ValidatedAndroidResources validated =
        merged.validate(AndroidDataContext.forNative(ruleContext));

    // Inherited values should be equal
    assertThat(merged).isEqualTo(new MergedAndroidResources(validated));

    // aapt artifacts should be generated
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.of(validated.getCompiledSymbols(), validated.getManifest()),
        /* outputs = */ ImmutableList.of(
            validated.getRTxt(), validated.getJavaSourceJar(), validated.getApk()));

    // aapt2 artifacts should be recorded
    assertThat(validated.getCompiledSymbols()).isNotNull();
    assertThat(validated.getAapt2RTxt()).isNotNull();
    assertThat(validated.getAapt2SourceJar()).isNotNull();
    assertThat(validated.getStaticLibrary()).isNotNull();

    // Compile the resources into compiled symbols files
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ validated.getResources(),
        /* outputs = */ ImmutableList.of(validated.getCompiledSymbols()));

    // Use the compiled symbols and manifest to build aapt2 packaging outputs
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.of(validated.getCompiledSymbols(), validated.getManifest()),
        /* outputs = */ ImmutableList.of(
            validated.getAapt2RTxt(), validated.getAapt2SourceJar(), validated.getStaticLibrary()));
  }

  @Test
  public void testValidate() throws Exception {
    RuleContext ruleContext = getRuleContext();

    makeMergedResources(ruleContext).validate(AndroidDataContext.forNative(ruleContext));

    Set<String> actionMnemonics =
        ruleContext.getAnalysisEnvironment().getRegisteredActions().stream()
            .map(ActionAnalysisMetadata::getMnemonic)
            .collect(ImmutableSet.toImmutableSet());
    // These are unfortunately the mnemonics used in Bazel; these should be changed once aapt1 is
    // removed.
    assertThat(actionMnemonics).contains("AndroidResourceLink"); // aapt2 validation
    assertThat(actionMnemonics).doesNotContain("AndroidResourceValidator"); // aapt1 validation
  }

  @Test
  public void testGenerateRClass() throws Exception {
    RuleContext ruleContext = getRuleContext();
    Artifact rTxt = ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT);
    ProcessedAndroidManifest manifest = getManifest();

    ProcessedAndroidData processedData =
        ProcessedAndroidData.of(
            makeParsedResources(ruleContext),
            AndroidAssets.from(ruleContext)
                .process(AndroidDataContext.forNative(ruleContext), AssetDependencies.empty()),
            manifest,
            rTxt,
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
            /* dataBindingInfoZip = */ null,
            ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
            null,
            null);

    ValidatedAndroidResources validated =
        processedData
            .generateRClass(AndroidDataContext.forNative(ruleContext))
            .getValidatedResources();

    // An action to generate the R.class file should be registered.
    assertActionArtifacts(
        ruleContext,
        /* inputs = */ ImmutableList.of(rTxt, manifest.getManifest()),
        /* outputs = */ ImmutableList.of(validated.getJavaClassJar()));
  }

  @Test
  public void testProcessBinaryDataGeneratesProguardOutput() throws Exception {
    RuleContext ruleContext = getRuleContext("android_binary", "manifest='AndroidManifest.xml',");
    AndroidDataContext dataContext = AndroidDataContext.forNative(ruleContext);

    ResourceApk resourceApk =
        ProcessedAndroidData.processBinaryDataFrom(
                dataContext,
                ruleContext,
                getManifest(),
                false,
                ImmutableMap.of(),
                AndroidResources.empty(),
                AndroidAssets.empty(),
                ResourceDependencies.empty(),
                AssetDependencies.empty(),
                ResourceFilterFactory.empty(),
                ImmutableList.of(),
                false,
                DataBinding.contextFrom(ruleContext, dataContext.getAndroidConfig()))
            .generateRClass(dataContext);

    assertThat(resourceApk.getResourceProguardConfig()).isNotNull();
    assertThat(resourceApk.getMainDexProguardConfig()).isNotNull();
  }

  /**
   * Validates that a parse action was invoked correctly. Returns the {@link ParsedAndroidResources}
   * for further validation.
   */
  private ParsedAndroidResources assertParse(RuleContext ruleContext) throws Exception {
    return assertParse(
        ruleContext,
        DataBinding.contextFrom(
            ruleContext, ruleContext.getConfiguration().getFragment(AndroidConfiguration.class)));
  }

  private ParsedAndroidResources assertParse(
      RuleContext ruleContext, DataBindingContext dataBindingContext) throws Exception {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "drawable-hdpi/bar.png");
    AndroidResources raw =
        new AndroidResources(
            resources, AndroidResources.getResourceRoots(ruleContext, resources, "resource_files"));
    StampedAndroidManifest manifest = getManifest();

    ParsedAndroidResources parsed =
        raw.parse(
            AndroidDataContext.forNative(ruleContext),
            manifest,
            dataBindingContext);

    // Inherited values should be equal
    assertThat(raw).isEqualTo(new AndroidResources(parsed));

    // Label should be set from RuleContext
    assertThat(parsed.getLabel()).isEqualTo(ruleContext.getLabel());

    return parsed;
  }

  private MergedAndroidResources makeMergedResources(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    return makeParsedResources(ruleContext)
        .merge(
            AndroidDataContext.forNative(ruleContext),
            ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false));
  }

  private ParsedAndroidResources makeParsedResources(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    DataBindingContext dataBindingContext =
        DataBinding.contextFrom(
            ruleContext, ruleContext.getConfiguration().getFragment(AndroidConfiguration.class));
    return makeParsedResources(ruleContext, dataBindingContext);
  }

  private ParsedAndroidResources makeParsedResources(
      RuleContext ruleContext, DataBindingContext dataBindingContext)
      throws RuleErrorException, InterruptedException {
    ImmutableList<Artifact> resources = getResources("values-en/foo.xml", "drawable-hdpi/bar.png");
    return new AndroidResources(
            resources, AndroidResources.getResourceRoots(ruleContext, resources, "resource_files"))
        .parse(
            AndroidDataContext.forNative(ruleContext),
            getManifest(),
            dataBindingContext);
  }

  private ProcessedAndroidManifest getManifest() {
    return new ProcessedAndroidManifest(
        getResource("some/path/AndroidManifest.xml"), "some.java.pkg", /* exported = */ true);
  }

  /** Gets a dummy rule context object by creating a dummy target. */
  private RuleContext getRuleContext() throws Exception {
    return getRuleContext("android_library");
  }

  private RuleContext getRuleContextWithDataBinding() throws Exception {
    return getRuleContext("android_library", "enable_data_binding = 1");
  }

  /** Gets a dummy rule context object by creating a dummy target. */
  private RuleContext getRuleContext(String kind, String... additionalLines) throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "java/foo",
            "target",
            ImmutableList.<String>builder()
                .add(kind + "(name = 'target',")
                .add(additionalLines)
                .add(")")
                .build()
                .toArray(new String[0]));
    return getRuleContextForActionTesting(target);
  }
}
