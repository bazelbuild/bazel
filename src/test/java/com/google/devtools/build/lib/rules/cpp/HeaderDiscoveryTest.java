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

import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
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
  private final ArtifactRoot artifactRoot =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, DERIVED_SEGMENT);

  @Test
  public void errorsWhenMissingHeaders() {
    ArtifactResolver artifactResolver = mock(ArtifactResolver.class);

    assertThrows(
        ActionExecutionException.class,
        () ->
            checkHeaderInclusion(
                artifactResolver,
                ImmutableList.of(
                    derivedRoot.getRelative("tree_artifact1/foo.h"),
                    derivedRoot.getRelative("tree_artifact1/subdir/foo.h")),
                NestedSetBuilder.create(
                    Order.STABLE_ORDER, treeArtifact(derivedRoot.getRelative("tree_artifact2")))));
  }

  private void checkHeaderInclusion(
      ArtifactResolver artifactResolver,
      ImmutableList<Path> dependencies,
      NestedSet<Artifact> includedHeaders)
      throws ActionExecutionException {
    var unused =
        HeaderDiscovery.discoverInputsFromDependencies(
            new ActionsTestUtil.NullAction(),
            ActionsTestUtil.createArtifact(artifactRoot, derivedRoot.getRelative("foo.cc")),
            /* shouldValidateInclusions= */ true,
            dependencies,
            /* permittedSystemIncludePrefixes= */ ImmutableList.of(),
            includedHeaders,
            execRoot,
            artifactResolver,
            /* siblingRepositoryLayout= */ false,
            PathMapper.NOOP);
  }

  private SpecialArtifact treeArtifact(Path path) {
    return SpecialArtifact.create(
        artifactRoot,
        artifactRoot.getExecPath().getRelative(artifactRoot.getRoot().relativize(path)),
        ActionsTestUtil.NULL_ARTIFACT_OWNER,
        Artifact.SpecialArtifactType.TREE);
  }
}
