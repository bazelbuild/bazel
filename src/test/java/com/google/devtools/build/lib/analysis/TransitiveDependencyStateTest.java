// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.getLast;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TransitiveDependencyStateTest {
  private static final Random rng = new Random(0);
  private static final Root fakeRoot =
      Root.fromPath(new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/fake"));

  @Test
  public void singlyAddedPackages_areSorted() {
    var orderedPackages =
        ImmutableList.<Package.Metadata>of(
            createFakePackageMetadata(PackageIdentifier.createInMainRepo("package1")),
            createFakePackageMetadata(PackageIdentifier.createInMainRepo("package2")),
            createFakePackageMetadata(PackageIdentifier.createInMainRepo("package3")));
    var workingCopy = new ArrayList<>(orderedPackages);

    for (int i = 0; i < 3; i++) {
      var state = newTransitiveState();

      Collections.shuffle(workingCopy, rng);
      workingCopy.forEach(state::updateTransitivePackages);

      assertThat(state.transitivePackages().toList())
          .containsExactlyElementsIn(orderedPackages)
          .inOrder();
    }
  }

  @Test
  public void configuredTargetPackages_areSorted() {
    ImmutableList<ConfiguredTargetKey> orderedKeys = getOrderedConfiguredTargetKeys();

    ImmutableList<Package.Metadata> orderedPackageMetadataList =
        createFakePackageMetadataList(orderedKeys.size());
    ImmutableList<NestedSet<Package.Metadata>> orderedPackageMetadataNestedSets =
        asSingletonNestedSets(orderedPackageMetadataList);

    var shuffledIndices = new ArrayList<Integer>();
    for (int i = 0; i < orderedKeys.size(); i++) {
      shuffledIndices.add(i);
    }

    for (int i = 0; i < 3; ++i) {
      var state = newTransitiveState();

      // Adds the entries to `state` in random order.
      Collections.shuffle(shuffledIndices, rng);
      for (int index : shuffledIndices) {
        state.updateTransitivePackages(
            orderedKeys.get(index), orderedPackageMetadataNestedSets.get(index));
      }

      // The result is always ordered.
      assertThat(state.transitivePackages().toList())
          .containsExactlyElementsIn(orderedPackageMetadataList)
          .inOrder();
    }
  }

  @Test
  public void aspectPackages_areSorted() {
    ImmutableList<AspectKey> orderedKeys = getOrderedAspectKeys();

    ImmutableList<Package.Metadata> orderedPackageMetadataList =
        createFakePackageMetadataList(orderedKeys.size());
    ImmutableList<NestedSet<Package.Metadata>> orderedPackagMetadataNestedSets =
        asSingletonNestedSets(orderedPackageMetadataList);

    var shuffledIndices = new ArrayList<Integer>();
    for (int i = 0; i < orderedKeys.size(); i++) {
      shuffledIndices.add(i);
    }

    for (int i = 0; i < 3; ++i) {
      var state = newTransitiveState();

      // Adds the entries to `state` in random order.
      Collections.shuffle(shuffledIndices, rng);
      for (int index : shuffledIndices) {
        state.updateTransitivePackages(
            orderedKeys.get(index), orderedPackagMetadataNestedSets.get(index));
      }

      // The result is always ordered.
      assertThat(state.transitivePackages().toList())
          .containsExactlyElementsIn(orderedPackageMetadataList)
          .inOrder();
    }
  }

  private static TransitiveDependencyState newTransitiveState() {
    return new TransitiveDependencyState(
        /* storeTransitivePackages= */ true, /* prerequisitePackages= */ p -> null);
  }

  private static Package.Metadata createFakePackageMetadata(PackageIdentifier id) {
    return Package.Metadata.builder()
        .packageIdentifier(id)
        .buildFilename(
            RootedPath.toRootedPath(
                fakeRoot, fakeRoot.getRelative(id.getPackageFragment().getRelative("BUILD"))))
        .workspaceName("workspace")
        .repositoryMapping(RepositoryMapping.EMPTY)
        .succinctTargetNotFoundErrors(PackageSettings.DEFAULTS.succinctTargetNotFoundErrors())
        .build();
  }

  private static ImmutableList<Package.Metadata> createFakePackageMetadataList(int count) {
    var orderedIds = new ArrayList<PackageIdentifier>(count);
    for (int i = 0; i < count; ++i) {
      orderedIds.add(PackageIdentifier.createInMainRepo("package" + i));
    }
    // Scrambles the order so if the result is ordered it's not somehow due to package sorting.
    Collections.shuffle(orderedIds, rng);
    return orderedIds.stream()
        .map(TransitiveDependencyStateTest::createFakePackageMetadata)
        .collect(toImmutableList());
  }

  private static ImmutableList<NestedSet<Package.Metadata>> asSingletonNestedSets(
      List<Package.Metadata> packageMetadataList) {
    return packageMetadataList.stream()
        .map(
            pkgMetadata ->
                NestedSetBuilder.<Package.Metadata>stableOrder().add(pkgMetadata).build())
        .collect(toImmutableList());
  }

  private static ImmutableSortedSet<BuildOptions> createTestOptions() {
    try {
      return ImmutableSortedSet.copyOf(
          comparing(BuildOptions::checksum),
          Arrays.<BuildOptions>asList(
              createTestOptions(ImmutableList.of("--platforms=" + TestConstants.PLATFORM_LABEL)),
              createTestOptions(ImmutableList.of("--platforms=" + MockObjcSupport.DARWIN_X86_64))));
    } catch (OptionsParsingException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static BuildOptions createTestOptions(List<String> args) throws OptionsParsingException {
    var fragments = ImmutableList.<Class<? extends FragmentOptions>>of(PlatformOptions.class);
    var optionsParser = OptionsParser.builder().optionsClasses(fragments).build();
    optionsParser.parse(args);
    return BuildOptions.of(fragments, optionsParser);
  }

  private static final ImmutableSortedSet<BuildOptions> TEST_OPTIONS = createTestOptions();
  private static final BuildOptions FIRST_OPTIONS = TEST_OPTIONS.iterator().next();
  private static final BuildOptions SECOND_OPTIONS = getLast(TEST_OPTIONS);

  private static ImmutableList<ConfiguredTargetKey> getOrderedConfiguredTargetKeys() {
    var label1 = Label.parseCanonicalUnchecked("//label1");
    var label2 = Label.parseCanonicalUnchecked("//label2");
    var platformLabel = Label.parseCanonicalUnchecked("//platforms:a");
    return ImmutableList.<ConfiguredTargetKey>of(
        ConfiguredTargetKey.builder().setLabel(label1).build(),
        ConfiguredTargetKey.builder()
            .setLabel(label1)
            .setConfigurationKey(BuildConfigurationKey.create(FIRST_OPTIONS))
            .build(),
        ConfiguredTargetKey.builder()
            .setLabel(label1)
            .setConfigurationKey(BuildConfigurationKey.create(SECOND_OPTIONS))
            .build(),
        ConfiguredTargetKey.builder()
            .setLabel(label1)
            .setExecutionPlatformLabel(platformLabel)
            .build(),
        ConfiguredTargetKey.builder()
            .setLabel(label1)
            .setExecutionPlatformLabel(platformLabel)
            .setConfigurationKey(BuildConfigurationKey.create(FIRST_OPTIONS))
            .build(),
        ConfiguredTargetKey.builder()
            .setLabel(label1)
            .setExecutionPlatformLabel(platformLabel)
            .setConfigurationKey(BuildConfigurationKey.create(SECOND_OPTIONS))
            .build(),
        ConfiguredTargetKey.builder().setLabel(label2).build());
  }

  private static final AspectClass ASPECT_CLASS1 = () -> "aspect1";
  private static final AspectClass ASPECT_CLASS2 = () -> "aspect2";
  private static final AspectClass ASPECT_CLASS3 = () -> "aspect3";
  private static final AspectClass ASPECT_CLASS4 = () -> "aspect4";

  private static ImmutableList<AspectDescriptor> getOrderedAspectDescriptors() {
    return ImmutableList.of(
        AspectDescriptor.of(ASPECT_CLASS1, AspectParameters.EMPTY),
        AspectDescriptor.of(
            ASPECT_CLASS1, new AspectParameters.Builder().addAttribute("foo", "bar").build()),
        AspectDescriptor.of(ASPECT_CLASS2, AspectParameters.EMPTY));
  }

  private static ImmutableList<AspectKey> getOrderedAspectKeys() {
    var descriptors = getOrderedAspectDescriptors();
    var builder = ImmutableList.<AspectKey>builder();

    var baseDescriptor1 = AspectDescriptor.of(ASPECT_CLASS3, AspectParameters.EMPTY);
    var baseDescriptor2 = AspectDescriptor.of(ASPECT_CLASS4, AspectParameters.EMPTY);

    for (var baseConfiguredTargetKey : getOrderedConfiguredTargetKeys()) {
      for (var descriptor : descriptors) {
        builder.add(AspectKeyCreator.createAspectKey(descriptor, baseConfiguredTargetKey));
      }

      // Constructs some additional keys that differ only in graph structure.
      var baseKey1 = AspectKeyCreator.createAspectKey(baseDescriptor1, baseConfiguredTargetKey);
      var baseKey2 = AspectKeyCreator.createAspectKey(baseDescriptor2, baseConfiguredTargetKey);

      builder.add(
          AspectKeyCreator.createAspectKey(
              getLast(descriptors), ImmutableList.of(baseKey1), baseConfiguredTargetKey));
      builder.add(
          AspectKeyCreator.createAspectKey(
              getLast(descriptors), ImmutableList.of(baseKey1, baseKey2), baseConfiguredTargetKey));
    }

    return builder.build();
  }
}
