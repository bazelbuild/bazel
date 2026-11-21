// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.constraints.ConstraintConstants;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.List;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test. */
@RunWith(JUnit4.class)
public final class HeaderDiscoveryTest {
  private static final String DERIVED_SEGMENT = "derived";

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/execroot");
  private final Path derivedRoot = execRoot.getChild(DERIVED_SEGMENT);
  private final ArtifactRoot derivedArtifactRoot =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, DERIVED_SEGMENT);
  private final ArtifactRoot sourceRoot = ArtifactRoot.asSourceRoot(Root.fromPath(execRoot));

  @Test
  public void errorsWhenMissingHeaders() {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of());

    assertThrows(
        ActionExecutionException.class,
        () ->
            discoverInputs(
                new ActionsTestUtil.NullAction(),
                artifactResolver,
                ImmutableList.of(
                    PathFragment.create(DERIVED_SEGMENT + "/tree_artifact1/foo.h"),
                    PathFragment.create(DERIVED_SEGMENT + "/tree_artifact1/subdir/foo.h")),
                /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
                NestedSetBuilder.create(
                    Order.STABLE_ORDER, treeArtifact(derivedRoot.getRelative("tree_artifact2")))));
  }

  @Test
  public void windowsPlatform_usesAsciiCaseInsensitiveResolution() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact resolvedArtifact = sourceArtifact("pkg/Include/Header.h");
    PathFragment depPath = PathFragment.create("pkg/include/header.h");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(
            eq(depPath), eq(RepositoryName.MAIN)))
        .thenReturn(ImmutableList.of(resolvedArtifact));

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/include/header.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    assertThat(result.toList()).containsExactly(resolvedArtifact);
    verify(artifactResolver, never()).resolveSourceArtifact(any(), any());
  }

  @Test
  public void nonWindowsPlatform_usesExactCaseResolution() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact resolvedArtifact = sourceArtifact("pkg/header.h");
    PathFragment depPath = PathFragment.create("pkg/header.h");
    when(artifactResolver.resolveSourceArtifact(eq(depPath), eq(RepositoryName.MAIN)))
        .thenReturn(resolvedArtifact);

    NestedSet<Artifact> result =
        discoverInputs(
            nonWindowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/header.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    assertThat(result.toList()).containsExactly(resolvedArtifact);
    verify(artifactResolver, never()).resolveSourceArtifactsAsciiCaseInsensitively(any(), any());
  }

  @Test
  public void windowsPlatform_singleCaseInsensitiveMatch_addsToInputs() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact resolvedArtifact = sourceArtifact("pkg/Include/BaseTsd.h");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of(resolvedArtifact));

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/include/basetsd.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    assertThat(result.toList()).containsExactly(resolvedArtifact);
  }

  @Test
  public void windowsPlatform_multipleCaseInsensitiveMatches_prefersActionInputs()
      throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact declaredInput = sourceArtifact("pkg/Include/Header.h");
    SourceArtifact otherVariant = sourceArtifact("pkg/include/header.h");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of(declaredInput, otherVariant));

    // The action has declaredInput in its inputs.
    NestedSet<Artifact> actionInputs =
        NestedSetBuilder.<Artifact>stableOrder().add(declaredInput).build();

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(actionInputs),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/include/header.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    // Only the variant that matches the declared input should be used.
    assertThat(result.toList()).containsExactly(declaredInput);
  }

  @Test
  public void windowsPlatform_multipleCaseInsensitiveMatches_noActionInputMatch_addsAll()
      throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact variant1 = sourceArtifact("pkg/Include/Header.h");
    SourceArtifact variant2 = sourceArtifact("pkg/include/header.h");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of(variant1, variant2));

    // No matching source artifacts in the action inputs.
    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/INCLUDE/HEADER.H")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    // When none of the matches are in the action inputs, all variants are added.
    assertThat(result.toList()).containsExactly(variant1, variant2);
  }

  @Test
  public void windowsPlatform_absoluteSystemInclude_matchesCaseInsensitively() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of());

    Path systemIncludeDir = fs.getPath("/C/Program Files/MSVC/include");
    // Compiler reports the path with different casing than the toolchain lists it.
    Path dep = fs.getPath("/c/program files/msvc/include/windows.h");

    // Should not throw — the absolute path should be filtered out as a system include.
    NestedSet<Artifact> result =
        HeaderDiscovery.discoverInputsFromDependencies(
            windowsAction(),
            ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("foo.cc")),
            /* shouldValidateInclusions= */ true,
            ImmutableList.of(dep.asFragment()),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(systemIncludeDir.asFragment()),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            execRoot,
            artifactResolver,
            PathMapper.NOOP);

    assertThat(result.toList()).isEmpty();
  }

  @Test
  public void windowsPlatform_absoluteSystemInclude_exactCaseAlsoMatches() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of());

    Path systemIncludeDir = fs.getPath("/C/Program Files/MSVC/include");
    Path dep = fs.getPath("/C/Program Files/MSVC/include/windows.h");

    NestedSet<Artifact> result =
        HeaderDiscovery.discoverInputsFromDependencies(
            windowsAction(),
            ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("foo.cc")),
            /* shouldValidateInclusions= */ true,
            ImmutableList.of(dep.asFragment()),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(systemIncludeDir.asFragment()),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            execRoot,
            artifactResolver,
            PathMapper.NOOP);

    assertThat(result.toList()).isEmpty();
  }

  @Test
  public void windowsPlatform_sourceFileFilteredOut() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact sourceFile = sourceArtifact("pkg/foo.cc");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of(sourceFile));

    NestedSet<Artifact> result =
        HeaderDiscovery.discoverInputsFromDependencies(
            windowsAction(),
            sourceFile,
            /* shouldValidateInclusions= */ true,
            ImmutableList.of(execRoot.getRelative("pkg/foo.cc").asFragment()),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            execRoot,
            artifactResolver,
            PathMapper.NOOP);

    // The source file itself should be filtered out as it's a mandatory input.
    assertThat(result.toList()).isEmpty();
  }

  @Test
  public void windowsPlatform_derivedArtifactMatchedExactly() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    Artifact derivedArtifact =
        ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("gen/foo.h"));

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create(DERIVED_SEGMENT + "/gen/foo.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.create(Order.STABLE_ORDER, derivedArtifact));

    // Derived artifacts should be matched by exact path, not going through
    // case-insensitive resolution.
    assertThat(result.toList()).containsExactly(derivedArtifact);
    verify(artifactResolver, never()).resolveSourceArtifactsAsciiCaseInsensitively(any(), any());
    verify(artifactResolver, never()).resolveSourceArtifact(any(), any());
  }

  @Test
  public void windowsPlatform_unresolvedSourcePath_errors() {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of());

    assertThrows(
        ActionExecutionException.class,
        () ->
            discoverInputs(
                windowsAction(),
                artifactResolver,
                ImmutableList.of(PathFragment.create("pkg/missing.h")),
                /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
                NestedSetBuilder.emptySet(Order.STABLE_ORDER)));
  }

  @Test
  public void windowsPlatform_multipleDeps_mixedResolution() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact sourceHeader = sourceArtifact("pkg/source.h");
    Artifact derivedHeader =
        ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("gen/gen.h"));

    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(
            eq(PathFragment.create("pkg/source.h")), any()))
        .thenReturn(ImmutableList.of(sourceHeader));

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(
                PathFragment.create("pkg/source.h"),
                PathFragment.create(DERIVED_SEGMENT + "/gen/gen.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.create(Order.STABLE_ORDER, derivedHeader));

    assertThat(result.toList()).containsExactly(sourceHeader, derivedHeader);
  }

  @Test
  public void windowsPlatform_multipleCaseInsensitiveMatches_onlyOneInActionInputs()
      throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact variant1 = sourceArtifact("pkg/Header.h");
    SourceArtifact variant2 = sourceArtifact("pkg/header.h");
    SourceArtifact variant3 = sourceArtifact("pkg/HEADER.h");
    when(artifactResolver.resolveSourceArtifactsAsciiCaseInsensitively(any(), any()))
        .thenReturn(ImmutableList.of(variant1, variant2, variant3));

    NestedSet<Artifact> actionInputs =
        NestedSetBuilder.<Artifact>stableOrder().add(variant2).build();

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(actionInputs),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/HEADER.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    // Only variant2 is in the action inputs, so only it should be selected.
    assertThat(result.toList()).containsExactly(variant2);
  }

  @Test
  public void relativeDependencies_resolvedAsExecPaths() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    SourceArtifact sourceHeader = sourceArtifact("pkg/header.h");
    Artifact derivedHeader =
        ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("gen/foo.h"));
    when(artifactResolver.resolveSourceArtifact(
            eq(PathFragment.create("pkg/header.h")), eq(RepositoryName.MAIN)))
        .thenReturn(sourceHeader);

    NestedSet<Artifact> result =
        discoverInputs(
            nonWindowsAction(),
            artifactResolver,
            ImmutableList.of(
                PathFragment.create("pkg/header.h"),
                PathFragment.create(DERIVED_SEGMENT + "/gen/foo.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(
                PathFragment.create("/usr/include")),
            NestedSetBuilder.create(Order.STABLE_ORDER, derivedHeader));

    assertThat(result.toList()).containsExactly(sourceHeader, derivedHeader);
  }

  @Test
  public void derivedArtifacts_onlyUsedOnesAreAdded() throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);
    Artifact usedHeader =
        ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("gen/used.h"));
    Artifact unusedHeader =
        ActionsTestUtil.createArtifact(
            derivedArtifactRoot, derivedRoot.getRelative("gen/unused.h"));

    NestedSet<Artifact> result =
        discoverInputs(
            nonWindowsAction(),
            artifactResolver,
            ImmutableList.of(
                PathFragment.create(DERIVED_SEGMENT + "/gen/used.h"),
                PathFragment.create(DERIVED_SEGMENT + "/gen/used.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            NestedSetBuilder.create(Order.STABLE_ORDER, unusedHeader, usedHeader));

    assertThat(result.toList()).containsExactly(usedHeader);
    verify(artifactResolver, never()).resolveSourceArtifact(any(), any());
  }

  @Test
  public void relativeDependency_belowSystemIncludePrefixBelowExecRoot_isIgnored()
      throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);

    NestedSet<Artifact> result =
        discoverInputs(
            nonWindowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("sysroot/include/stdio.h")),
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(
                execRoot.getRelative("sysroot/include").asFragment()),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    assertThat(result.toList()).isEmpty();
    verify(artifactResolver, never()).resolveSourceArtifact(any(), any());
  }

  @Test
  public void relativeDependency_belowSystemIncludePrefixAboveExecRoot_isIgnored()
      throws Exception {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);

    NestedSet<Artifact> result =
        discoverInputs(
            windowsAction(),
            artifactResolver,
            ImmutableList.of(PathFragment.create("pkg/header.h")),
            // The exec root is "/execroot", but this is a case-insensitive file system.
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(
                PathFragment.create("/EXECROOT")),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    assertThat(result.toList()).isEmpty();
    verify(artifactResolver, never()).resolveSourceArtifactsAsciiCaseInsensitively(any(), any());
  }

  @Test
  public void absoluteDependency_outsideOfExecRootAndSystemIncludePrefixes_errors() {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);

    assertThrows(
        ActionExecutionException.class,
        () ->
            discoverInputs(
                nonWindowsAction(),
                artifactResolver,
                ImmutableList.of(PathFragment.create("/opt/include/header.h")),
                /* permittedSystemIncludePrefixes= */ ImmutableList.of(
                    PathFragment.create("/usr/include")),
                NestedSetBuilder.emptySet(Order.STABLE_ORDER)));
  }

  // Helpers

  private NestedSet<Artifact> discoverInputs(
      ActionsTestUtil.NullAction action,
      ArtifactResolver artifactResolver,
      List<PathFragment> dependencies,
      List<PathFragment> permittedSystemIncludePrefixes,
      NestedSet<Artifact> allowedDerivedInputs)
      throws ActionExecutionException {
    return HeaderDiscovery.discoverInputsFromDependencies(
        action,
        ActionsTestUtil.createArtifact(derivedArtifactRoot, derivedRoot.getRelative("foo.cc")),
        /* shouldValidateInclusions= */ true,
        dependencies,
        permittedSystemIncludePrefixes,
        allowedDerivedInputs,
        execRoot,
        artifactResolver,
        PathMapper.NOOP);
  }

  private TreeArtifact treeArtifact(Path path) {
    return TreeArtifact.create(
        derivedArtifactRoot,
        derivedArtifactRoot
            .getExecPath()
            .getRelative(derivedArtifactRoot.getRoot().relativize(path)),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
  }

  private SourceArtifact sourceArtifact(String execPath) {
    return new SourceArtifact(sourceRoot, PathFragment.create(execPath), ArtifactOwner.NULL_OWNER);
  }

  private static PlatformInfo windowsPlatform() {
    try {
      return PlatformInfo.builder()
          .setLabel(Label.parseCanonicalUnchecked("//test:windows_platform"))
          .addConstraint(ConstraintConstants.OS_TO_DEFAULT_CONSTRAINT_VALUE.get(OS.WINDOWS))
          .build();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static PlatformInfo linuxPlatform() {
    try {
      return PlatformInfo.builder()
          .setLabel(Label.parseCanonicalUnchecked("//test:linux_platform"))
          .addConstraint(ConstraintConstants.OS_TO_DEFAULT_CONSTRAINT_VALUE.get(OS.LINUX))
          .build();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static ActionOwner actionOwnerWithPlatform(PlatformInfo platform) {
    return ActionOwner.createDummy(
        Label.parseCanonicalUnchecked("//pkg:target"),
        new Location("dummy-file", 0, 0),
        /* targetKind= */ "cc_library rule",
        /* buildConfigurationMnemonic= */ "k8-fastbuild",
        /* configurationChecksum= */ "checksum",
        /* buildConfigurationEvent= */ null,
        /* isToolConfiguration= */ false,
        platform,
        /* aspectDescriptors= */ ImmutableList.of(),
        /* execProperties= */ ImmutableMap.of());
  }

  private ActionsTestUtil.NullAction windowsAction() {
    return new ActionsTestUtil.NullAction(
        actionOwnerWithPlatform(windowsPlatform()), ActionsTestUtil.DUMMY_ARTIFACT);
  }

  private ActionsTestUtil.NullAction windowsAction(NestedSet<Artifact> inputs) {
    return new ActionsTestUtil.NullAction(actionOwnerWithPlatform(windowsPlatform()), inputs);
  }

  private ActionsTestUtil.NullAction nonWindowsAction() {
    return new ActionsTestUtil.NullAction(
        actionOwnerWithPlatform(linuxPlatform()), ActionsTestUtil.DUMMY_ARTIFACT);
  }
}
