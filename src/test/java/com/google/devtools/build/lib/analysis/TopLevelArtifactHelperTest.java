// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX;
import static com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.getAllArtifactsToBuild;
import static java.util.Arrays.asList;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TopLevelArtifactHelper}. */
@RunWith(JUnit4.class)
public class TopLevelArtifactHelperTest {

  private TopLevelArtifactContext ctx;
  private OutputGroupInfo groupProvider;

  private Path path;
  private ArtifactRoot root;
  private int artifactIdx;

  @Before
  public final void setRootDir() throws Exception {
    Scratch scratch = new Scratch();
    Path execRoot = scratch.getFileSystem().getPath("/");
    root = ArtifactRoot.asDerivedRoot(execRoot, "blaze-out");
    path = scratch.dir("/blaze-out/foo");
  }

  private void setup(Iterable<Pair<String, Integer>> groupArtifacts) {
    ImmutableSortedSet.Builder<String> setBuilder = ImmutableSortedSet.naturalOrder();
    ImmutableMap.Builder<String, NestedSet<Artifact>> mapBuilder = ImmutableMap.builder();
    for (Pair<String, Integer> groupArtifact : groupArtifacts) {
      setBuilder.add(checkNotNull(groupArtifact.getFirst()));
      mapBuilder.put(
          groupArtifact.getFirst(), newArtifacts(checkNotNull(groupArtifact.getSecond())));
    }
    ctx = new TopLevelArtifactContext(false, false, false, setBuilder.build());
    groupProvider = new OutputGroupInfo(mapBuilder.build());
  }

  @Test
  public void artifactsShouldBeSeparateByGroup() {
    setup(asList(Pair.of("foo", 3), Pair.of("bar", 2)));

    ArtifactsToBuild allArtifacts = getAllArtifactsToBuild(groupProvider, null, ctx);
    assertThat(allArtifacts.getAllArtifacts().toList()).hasSize(5);
    assertThat(allArtifacts.getImportantArtifacts().toList()).hasSize(5);

    NestedSet<ArtifactsInOutputGroup> artifactsByGroup =
        allArtifacts.getAllArtifactsByOutputGroup();
    // Two groups
    assertThat(artifactsByGroup.toList()).hasSize(2);

    for (ArtifactsInOutputGroup artifacts : artifactsByGroup.toList()) {
      String outputGroup = artifacts.getOutputGroup();
      if ("foo".equals(outputGroup)) {
        assertThat(artifacts.getArtifacts().toList()).hasSize(3);
      } else if ("bar".equals(outputGroup)) {
        assertThat(artifacts.getArtifacts().toList()).hasSize(2);
      }
    }
  }

  @Test
  public void emptyGroupsShouldBeIgnored() {
    setup(asList(Pair.of("foo", 1), Pair.of("bar", 0)));

    ArtifactsToBuild allArtifacts = getAllArtifactsToBuild(groupProvider, null, ctx);
    assertThat(allArtifacts.getAllArtifacts().toList()).hasSize(1);
    assertThat(allArtifacts.getImportantArtifacts().toList()).hasSize(1);

    NestedSet<ArtifactsInOutputGroup> artifactsByGroup =
        allArtifacts.getAllArtifactsByOutputGroup();
    // The bar list should not appear here, as it contains no artifacts.
    assertThat(artifactsByGroup.toList()).hasSize(1);
    assertThat(artifactsByGroup.toList().get(0).getOutputGroup()).isEqualTo("foo");
  }

  @Test
  public void importantArtifacts() {
    setup(asList(Pair.of(HIDDEN_OUTPUT_GROUP_PREFIX + "notimportant", 1), Pair.of("important", 2)));

    ArtifactsToBuild allArtifacts = getAllArtifactsToBuild(groupProvider, null, ctx);
    assertThat(allArtifacts.getAllArtifacts().toList()).hasSize(3);
    assertThat(allArtifacts.getImportantArtifacts().toList()).hasSize(2);
  }

  private NestedSet<Artifact> newArtifacts(int num) {
    NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(Order.STABLE_ORDER);
    for (int i = 0; i < num; i++) {
      builder.add(newArtifact());
    }
    return builder.build();
  }

  private Artifact newArtifact() {
    return ActionsTestUtil.createArtifact(root, path.getRelative(Integer.toString(artifactIdx++)));
  }
}
