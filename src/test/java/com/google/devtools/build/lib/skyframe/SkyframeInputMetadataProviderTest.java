// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MissingDepExecException;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SkyframeInputMetadataProviderTest extends FoundationTestCase {
  // The behavior this test verifies (that SkyValues are memoized over multiple restarts) is not
  // actually necessary for the SkyframeInputMetadataProvider, but only due to a pretty brittle
  // combination of happenstances: action rewinding may remove a value from the graph at any time
  // and then MemoizingEvaluator.getExistingValue() will return null. However, Skyframe stores
  // previously requested direct dependencies in SkyFunction.Environment, so when that happens,
  // the requested metadata is still returned. But this relies on the particular implementation of
  // Skyframe *and* SkyframeInputMetadataProvider and memoizing over restarts isn't that costly so
  // it's useful to signal to someone who would remove this memoization, either accidentally or
  // intentionally.
  @Test
  public void skyframeLookupsMemoizedOverMultipleRestarts() throws Exception {
    StaticInputMetadataProvider perBuild = new StaticInputMetadataProvider(ImmutableMap.of());

    ActionLookupKey owner = ActionsTestUtil.createActionLookupKey("owner");
    ActionLookupData actionKey = ActionLookupData.create(owner, 0);
    DerivedArtifact artifact =
        DerivedArtifact.create(
            ArtifactRoot.asDerivedRoot(root.asPath(), RootType.OUTPUT, "out"),
            PathFragment.create("out/foo"),
            owner);
    artifact.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(artifact.getPath(), "test");
    artifact.setGeneratingActionKey(actionKey);

    MemoizingEvaluator evaluator = mock(MemoizingEvaluator.class);
    SkyframeInputMetadataProvider simp =
        new SkyframeInputMetadataProvider(evaluator, perBuild, "out");

    // On the first iteration, the dependency is not available yet. getInputMetadataChecked()
    // should accordingly throw.
    SkyFunction.Environment env1 = mock(SkyFunction.Environment.class);
    when(evaluator.getExistingValue(actionKey)).thenReturn(null);
    when(env1.getValue(actionKey)).thenReturn(null);
    try (var unused = simp.withSkyframeAllowed(env1)) {
      assertThrows(MissingDepExecException.class, () -> simp.getInputMetadataChecked(artifact));
    }

    FileArtifactValue metadata = FileArtifactValue.createForTesting(artifact);
    ActionExecutionValue aev =
        ActionExecutionValue.create(
            ImmutableMap.of(artifact, metadata),
            ImmutableMap.of(),
            null,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    // Now the artifact in in Skyframe. Its metadata should be returned.
    SkyFunction.Environment env2 = mock(SkyFunction.Environment.class);
    when(evaluator.getExistingValue(actionKey)).thenReturn(aev);
    when(env2.getValue(actionKey)).thenReturn(aev);
    try (var unused = simp.withSkyframeAllowed(env2)) {
      assertThat(simp.getInputMetadataChecked(artifact)).isEqualTo(metadata);
    }

    // No further methods on env3 or the evaluator should be called and the metadata should still be
    // returned as normal.
    when(evaluator.getExistingValue(actionKey)).thenThrow(IllegalStateException.class);
    SkyFunction.Environment env3 = mock(SkyFunction.Environment.class);
    try (var unused = simp.withSkyframeAllowed(env3)) {
      assertThat(simp.getInputMetadataChecked(artifact)).isEqualTo(metadata);
    }

    verify(env3, never()).getValue(actionKey);
  }

  @Test
  public void skyframeLookupsMemoizedWithinASingleRestart() throws Exception {
    StaticInputMetadataProvider perBuild = new StaticInputMetadataProvider(ImmutableMap.of());

    ActionLookupKey owner = ActionsTestUtil.createActionLookupKey("owner");
    ActionLookupData actionKey = ActionLookupData.create(owner, 0);
    DerivedArtifact artifact =
        DerivedArtifact.create(
            ArtifactRoot.asDerivedRoot(root.asPath(), RootType.OUTPUT, "out"),
            PathFragment.create("out/foo"),
            owner);
    artifact.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(artifact.getPath(), "test");
    artifact.setGeneratingActionKey(actionKey);
    FileArtifactValue metadata = FileArtifactValue.createForTesting(artifact);
    ActionExecutionValue aev =
        ActionExecutionValue.create(
            ImmutableMap.of(artifact, metadata),
            ImmutableMap.of(),
            null,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    MemoizingEvaluator evaluator = mock(MemoizingEvaluator.class);
    SkyframeInputMetadataProvider simp =
        new SkyframeInputMetadataProvider(evaluator, perBuild, "out");

    SkyFunction.Environment env = mock(SkyFunction.Environment.class);
    when(evaluator.getExistingValue(actionKey))
        .thenReturn(aev) // first call
        .thenThrow(IllegalStateException.class); // Subsequent calls

    try (var unused = simp.withSkyframeAllowed(env)) {
      assertThat(simp.getInputMetadataChecked(artifact)).isEqualTo(metadata);
      assertThat(simp.getInputMetadataChecked(artifact)).isEqualTo(metadata);
    }

    verify(env, never()).getValue(any());
  }
}
