// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root.RootCodecDependencies;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ActionExecutionValueTest {
  private static final FileArtifactValue VALUE_1_REMOTE =
      RemoteFileArtifactValue.create(
          /* digest= */ new byte[0],
          /* size= */ 0,
          /* locationIndex= */ 1,
          /* expireAtEpochMilli= */ -1);
  private static final FileArtifactValue VALUE_2_REMOTE =
      RemoteFileArtifactValue.create(
          /* digest= */ new byte[0],
          /* size= */ 0,
          /* locationIndex= */ 2,
          /* expireAtEpochMilli= */ -1);

  private static final ActionLookupKey KEY = ActionsTestUtil.NULL_ARTIFACT_OWNER;
  private static final ActionLookupData ACTION_LOOKUP_DATA_1 = ActionLookupData.create(KEY, 1);
  private static final ActionLookupData ACTION_LOOKUP_DATA_2 = ActionLookupData.create(KEY, 2);

  private static final ArtifactRoot OUTPUT_ROOT =
      ArtifactRoot.asDerivedRoot(new Scratch().resolve("/execroot"), RootType.Output, "out");
  private final Scratch scratch =
      new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256), "/root");

  @Test
  public void equality() {
    SpecialArtifact tree1 = tree("tree1");
    TreeArtifactValue tree1Value1 =
        TreeArtifactValue.newBuilder(tree1)
            .putChild(TreeFileArtifact.createTreeOutput(tree1, "file1"), VALUE_1_REMOTE)
            .build();
    TreeArtifactValue tree1Value2 =
        TreeArtifactValue.newBuilder(tree1)
            .putChild(TreeFileArtifact.createTreeOutput(tree1, "file1"), VALUE_2_REMOTE)
            .build();
    FilesetOutputSymlink symlink1 =
        FilesetOutputSymlink.createForTesting(
            PathFragment.create("name1"),
            PathFragment.create("target1"),
            PathFragment.create("execPath1"));
    FilesetOutputSymlink symlink2 =
        FilesetOutputSymlink.createForTesting(
            PathFragment.create("name2"),
            PathFragment.create("target2"),
            PathFragment.create("execPath2"));

    new EqualsTester()
        .addEqualityGroup(createWithArtifactData(ImmutableMap.of(output("file1"), VALUE_1_REMOTE)))
        .addEqualityGroup(createWithArtifactData(ImmutableMap.of(output("file1"), VALUE_2_REMOTE)))
        .addEqualityGroup(
            createWithArtifactData(
                ImmutableMap.of(output("file1", ACTION_LOOKUP_DATA_2), VALUE_1_REMOTE)))
        .addEqualityGroup(createWithArtifactData(ImmutableMap.of(output("file2"), VALUE_1_REMOTE)))
        .addEqualityGroup(
            createWithArtifactData(
                ImmutableMap.of(output("file1"), VALUE_1_REMOTE, output("file2"), VALUE_2_REMOTE)))
        // treeArtifactData
        .addEqualityGroup(
            createWithTreeArtifactData(ImmutableMap.of(tree1, TreeArtifactValue.empty())))
        .addEqualityGroup(
            createWithTreeArtifactData(ImmutableMap.of(tree("tree2"), TreeArtifactValue.empty())))
        .addEqualityGroup(
            ImmutableMap.of(
                tree1, TreeArtifactValue.empty(), tree("tree2"), TreeArtifactValue.empty()))
        .addEqualityGroup(ImmutableMap.of(tree1, tree1Value1))
        .addEqualityGroup(ImmutableMap.of(tree1, tree1Value2))
        // outputSymlinks
        .addEqualityGroup(createWithOutputSymlinks(ImmutableList.of(symlink1)))
        .addEqualityGroup(createWithOutputSymlinks(ImmutableList.of(symlink2)))
        .addEqualityGroup(createWithOutputSymlinks(ImmutableList.of(symlink1, symlink2)))
        .addEqualityGroup(createWithOutputSymlinks(ImmutableList.of(symlink2, symlink1)))
        // discoveredModules
        .addEqualityGroup(
            createWithDiscoveredModules(
                NestedSetBuilder.create(Order.STABLE_ORDER, output("file1"))),
            createWithDiscoveredModules(
                NestedSetBuilder.create(Order.STABLE_ORDER, output("file1"))))
        .addEqualityGroup(
            createWithDiscoveredModules(
                NestedSetBuilder.create(Order.STABLE_ORDER, output("file1", ACTION_LOOKUP_DATA_2))))
        .addEqualityGroup(
            createWithDiscoveredModules(
                NestedSetBuilder.<Artifact>stableOrder().add(output("file2"))))
        .addEqualityGroup(
            createWithDiscoveredModules(
                NestedSetBuilder.<Artifact>stableOrder().add(output("file1")).add(output("file2"))))
        // Does not detect equality for identical sets with different shape.
        .addEqualityGroup(
            createWithDiscoveredModules(
                NestedSetBuilder.<Artifact>stableOrder()
                    .add(output("file1"))
                    .addTransitive(NestedSetBuilder.create(Order.STABLE_ORDER, output("file2")))))
        .testEquals();
  }

  @Test
  public void serialization() throws Exception {
    new SerializationTester(
            // Single output file
            createWithArtifactData(ImmutableMap.of(output("output1"), VALUE_1_REMOTE)),
            // Fileset
            createWithOutputSymlinks(
                ImmutableList.of(
                    FilesetOutputSymlink.createForTesting(
                        PathFragment.create("name"),
                        PathFragment.create("target"),
                        PathFragment.create("execPath")))),
            // Module discovering
            createWithDiscoveredModules(
                NestedSetBuilder.create(Order.STABLE_ORDER, output("module"))),
            // Multiple output files
            createWithArtifactData(
                ImmutableMap.of(
                    output("output1"), VALUE_1_REMOTE, output("output2"), VALUE_2_REMOTE)),
            // Single tree
            createWithTreeArtifactData(ImmutableMap.of(tree("tree"), TreeArtifactValue.empty())),
            // Multiple trees
            createWithTreeArtifactData(
                ImmutableMap.of(
                    tree("tree1"),
                    TreeArtifactValue.empty(),
                    tree("tree2"),
                    TreeArtifactValue.empty())),
            // Mixed file and tree
            ActionExecutionValue.create(
                ImmutableMap.of(output("file"), VALUE_1_REMOTE),
                ImmutableMap.of(tree("tree"), TreeArtifactValue.empty()),
                /* outputSymlinks= */ ImmutableList.of(),
                /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER)))
        .addDependency(FileSystem.class, OUTPUT_ROOT.getRoot().getFileSystem())
        .addDependency(
            RootCodecDependencies.class, new RootCodecDependencies(OUTPUT_ROOT.getRoot()))
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .runTests();
  }

  @Test
  public void isEntirelyRemote() throws IOException {
    Path file1 = scratch.file("/file1");
    FileArtifactValue value1Local =
        FileArtifactValue.createFromStat(file1, file1.stat(Symlinks.FOLLOW), SyscallCache.NO_CACHE);

    // Remote artifact.
    ActionExecutionValue actionExecutionValue1 =
        createWithArtifactData(ImmutableMap.of(output("file1"), VALUE_1_REMOTE));

    assertThat(actionExecutionValue1.isEntirelyRemote()).isTrue();

    // Local artifact.
    ActionExecutionValue actionExecutionValue2 =
        createWithArtifactData(ImmutableMap.of(output("file1"), value1Local));

    assertThat(actionExecutionValue2.isEntirelyRemote()).isFalse();

    // Local and remote artifacts.
    ActionExecutionValue actionExecutionValue3 =
        createWithArtifactData(
            ImmutableMap.of(output("file1"), value1Local, output("file2"), VALUE_2_REMOTE));

    assertThat(actionExecutionValue3.isEntirelyRemote()).isFalse();

    SpecialArtifact tree1 = tree("tree1");
    TreeArtifactValue tree1Value1Remote =
        TreeArtifactValue.newBuilder(tree1)
            .putChild(TreeFileArtifact.createTreeOutput(tree1, "file1"), VALUE_1_REMOTE)
            .build();

    // Remote tree artifact.
    ActionExecutionValue actionExecutionValue4 =
        createWithTreeArtifactData(ImmutableMap.of(tree1, tree1Value1Remote));

    assertThat(actionExecutionValue4.isEntirelyRemote()).isTrue();

    SpecialArtifact tree2 = tree("tree2");
    Path file2 = scratch.file("/file2");
    FileArtifactValue value2Local =
        FileArtifactValue.createFromStat(file2, file2.stat(Symlinks.FOLLOW), SyscallCache.NO_CACHE);
    TreeArtifactValue tree2Value2Local =
        TreeArtifactValue.newBuilder(tree2)
            .putChild(TreeFileArtifact.createTreeOutput(tree2, "file2"), value2Local)
            .build();

    // Local tree artifact.
    ActionExecutionValue actionExecutionValue5 =
        createWithTreeArtifactData(ImmutableMap.of(tree2, tree2Value2Local));

    assertThat(actionExecutionValue5.isEntirelyRemote()).isFalse();

    // Local and remote tree artifacts.
    ActionExecutionValue actionExecutionValue6 =
        createWithTreeArtifactData(
            ImmutableMap.of(tree2, tree2Value2Local, tree1, tree1Value1Remote));

    assertThat(actionExecutionValue6.isEntirelyRemote()).isFalse();

    // Remote artifact and local tree artifact.
    ActionExecutionValue actionExecutionValue7 =
        createWithArtifactAndTreeArtifactData(
            ImmutableMap.of(output("file1"), VALUE_1_REMOTE),
            ImmutableMap.of(tree2, tree2Value2Local));

    assertThat(actionExecutionValue7.isEntirelyRemote()).isFalse();

    // Local artifact and remote tree artifact.
    ActionExecutionValue actionExecutionValue8 =
        createWithArtifactAndTreeArtifactData(
            ImmutableMap.of(output("file2"), value2Local),
            ImmutableMap.of(tree1, tree1Value1Remote));

    assertThat(actionExecutionValue8.isEntirelyRemote()).isFalse();

    // Local artifact and tree artifact.
    ActionExecutionValue actionExecutionValue9 =
        createWithArtifactAndTreeArtifactData(
            ImmutableMap.of(output("file1"), value1Local),
            ImmutableMap.of(tree2, tree2Value2Local));

    assertThat(actionExecutionValue9.isEntirelyRemote()).isFalse();

    // Remote artifact and tree artifact.
    ActionExecutionValue actionExecutionValue10 =
        createWithArtifactAndTreeArtifactData(
            ImmutableMap.of(output("file2"), VALUE_2_REMOTE),
            ImmutableMap.of(tree1, tree1Value1Remote));

    assertThat(actionExecutionValue10.isEntirelyRemote()).isTrue();

    // Empty tree artifact.
    ActionExecutionValue actionExecutionValue11 =
        createWithTreeArtifactData(ImmutableMap.of(tree1, TreeArtifactValue.empty()));

    assertThat(actionExecutionValue11.isEntirelyRemote()).isFalse();

    // Discovered modules.
    ActionExecutionValue actionExecutionValue12 =
        createWithDiscoveredModules(NestedSetBuilder.create(Order.STABLE_ORDER, output("file1")));

    assertThat(actionExecutionValue12.isEntirelyRemote()).isTrue();

    FilesetOutputSymlink symlink1 =
        FilesetOutputSymlink.createForTesting(
            PathFragment.create("name1"),
            PathFragment.create("target1"),
            PathFragment.create("execPath1"));

    // Fileset.
    ActionExecutionValue actionExecutionValue13 =
        createWithOutputSymlinks(ImmutableList.of(symlink1));

    assertThat(actionExecutionValue13.isEntirelyRemote()).isTrue();
  }

  private static ActionExecutionValue createWithArtifactData(
      ImmutableMap<Artifact, FileArtifactValue> artifactData) {
    return ActionExecutionValue.create(
        /* artifactData= */ artifactData,
        /* treeArtifactData= */ ImmutableMap.of(),
        /* outputSymlinks= */ ImmutableList.of(),
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static ActionExecutionValue createWithTreeArtifactData(
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData) {
    return ActionExecutionValue.create(
        /* artifactData= */ ImmutableMap.of(),
        treeArtifactData,
        /* outputSymlinks= */ ImmutableList.of(),
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static ActionExecutionValue createWithArtifactAndTreeArtifactData(
      ImmutableMap<Artifact, FileArtifactValue> artifactData,
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData) {
    return ActionExecutionValue.create(
        artifactData,
        treeArtifactData,
        /* outputSymlinks= */ ImmutableList.of(),
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static ActionExecutionValue createWithOutputSymlinks(
      ImmutableList<FilesetOutputSymlink> outputSymlinks) {
    return ActionExecutionValue.create(
        ImmutableMap.of(output("fileset.manifest"), VALUE_1_REMOTE),
        /* treeArtifactData= */ ImmutableMap.of(),
        outputSymlinks,
        /* discoveredModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  private static ActionExecutionValue createWithDiscoveredModules(
      NestedSetBuilder<Artifact> discoveredModules) {
    return createWithDiscoveredModules(discoveredModules.build());
  }

  private static ActionExecutionValue createWithDiscoveredModules(
      NestedSet<Artifact> discoveredModules) {
    return ActionExecutionValue.create(
        /* artifactData= */ ImmutableMap.of(output("modules.pcm"), VALUE_1_REMOTE),
        /* treeArtifactData= */ ImmutableMap.of(),
        /* outputSymlinks= */ ImmutableList.of(),
        discoveredModules);
  }

  private static DerivedArtifact output(String rootRelativePath) {
    return output(rootRelativePath, ACTION_LOOKUP_DATA_1);
  }

  private static DerivedArtifact output(
      String rootRelativePath, ActionLookupData generatingAction) {
    DerivedArtifact result =
        DerivedArtifact.create(
            OUTPUT_ROOT,
            OUTPUT_ROOT.getExecPath().getRelative(rootRelativePath),
            generatingAction.getActionLookupKey());
    result.setGeneratingActionKey(generatingAction);
    return result;
  }

  private static SpecialArtifact tree(String rootRelativePath) {
    SpecialArtifact result =
        SpecialArtifact.create(
            OUTPUT_ROOT,
            OUTPUT_ROOT.getExecPath().getRelative(rootRelativePath),
            KEY,
            SpecialArtifactType.TREE);
    result.setGeneratingActionKey(ACTION_LOOKUP_DATA_1);
    return result;
  }
}
