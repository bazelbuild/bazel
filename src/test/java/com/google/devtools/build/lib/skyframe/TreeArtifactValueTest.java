// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link TreeArtifactValue}. */
@RunWith(Parameterized.class)
public final class TreeArtifactValueTest {

  private static final ArtifactRoot ROOT =
      ArtifactRoot.asDerivedRoot(new InMemoryFileSystem().getPath("/root"), "bin");

  private enum MultiBuilderType {
    BASIC {
      @Override
      TreeArtifactValue.MultiBuilder newMultiBuilder() {
        return TreeArtifactValue.newMultiBuilder();
      }
    },
    CONCURRENT {
      @Override
      TreeArtifactValue.MultiBuilder newMultiBuilder() {
        return TreeArtifactValue.newConcurrentMultiBuilder();
      }
    };

    abstract TreeArtifactValue.MultiBuilder newMultiBuilder();
  }

  @Parameter public MultiBuilderType multiBuilderType;
  private final FakeMetadataInjector metadataInjector = new FakeMetadataInjector();

  @Parameters(name = "{0}")
  public static MultiBuilderType[] params() {
    return MultiBuilderType.values();
  }

  @Test
  public void empty() {
    TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();

    treeArtifacts.injectTo(metadataInjector);

    assertThat(metadataInjector.injectedTreeArtifacts).isEmpty();
  }

  @Test
  public void singleTreeArtifact() {
    TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();
    SpecialArtifact parent = createTreeArtifact("tree");
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(parent, "child1");
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(parent, "child2");

    treeArtifacts.putChild(child1, metadataWithId(1));
    treeArtifacts.putChild(child2, metadataWithId(2));
    treeArtifacts.injectTo(metadataInjector);

    assertThat(metadataInjector.injectedTreeArtifacts)
        .containsExactly(
            parent,
            TreeArtifactValue.create(
                ImmutableMap.of(child1, metadataWithId(1), child2, metadataWithId(2))));
  }

  @Test
  public void multipleTreeArtifacts() {
    TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();
    SpecialArtifact parent1 = createTreeArtifact("tree1");
    TreeFileArtifact parent1Child1 = TreeFileArtifact.createTreeOutput(parent1, "child1");
    TreeFileArtifact parent1Child2 = TreeFileArtifact.createTreeOutput(parent1, "child2");
    SpecialArtifact parent2 = createTreeArtifact("tree2");
    TreeFileArtifact parent2Child = TreeFileArtifact.createTreeOutput(parent2, "child");

    treeArtifacts.putChild(parent1Child1, metadataWithId(1));
    treeArtifacts.putChild(parent2Child, metadataWithId(3));
    treeArtifacts.putChild(parent1Child2, metadataWithId(2));
    treeArtifacts.injectTo(metadataInjector);

    assertThat(metadataInjector.injectedTreeArtifacts)
        .containsExactly(
            parent1,
            TreeArtifactValue.create(
                ImmutableMap.of(
                    parent1Child1, metadataWithId(1), parent1Child2, metadataWithId(2))),
            parent2,
            TreeArtifactValue.create(ImmutableMap.of(parent2Child, metadataWithId(3))));
  }

  private static SpecialArtifact createTreeArtifact(String execPath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        ROOT, PathFragment.create(execPath));
  }

  private static FileArtifactValue metadataWithId(int id) {
    return new RemoteFileArtifactValue(new byte[] {(byte) id}, id, id);
  }

  private static final class FakeMetadataInjector implements MetadataInjector {
    private final Map<SpecialArtifact, TreeArtifactValue> injectedTreeArtifacts = new HashMap<>();

    @Override
    public void injectFile(Artifact output, FileArtifactValue metadata) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void injectDirectory(
        SpecialArtifact output, Map<TreeFileArtifact, FileArtifactValue> children) {
      injectedTreeArtifacts.put(output, TreeArtifactValue.create(children));
    }

    @Override
    public void markOmitted(Artifact output) {
      throw new UnsupportedOperationException();
    }
  }
}
