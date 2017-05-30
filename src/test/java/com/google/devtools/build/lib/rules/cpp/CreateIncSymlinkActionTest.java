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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link CreateIncSymlinkAction}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class CreateIncSymlinkActionTest extends FoundationTestCase {

  @Test
  public void testDifferentOrderSameActionKey() throws Exception {
    Root root = Root.asDerivedRoot(rootDirectory, rootDirectory.getRelative("out"));
    Artifact a = new Artifact(PathFragment.create("a"), root);
    Artifact b = new Artifact(PathFragment.create("b"), root);
    Artifact c = new Artifact(PathFragment.create("c"), root);
    Artifact d = new Artifact(PathFragment.create("d"), root);
    CreateIncSymlinkAction action1 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b, c, d), root.getPath());
    // Can't reuse the artifacts here; that would lead to DuplicateArtifactException.
    a = new Artifact(PathFragment.create("a"), root);
    b = new Artifact(PathFragment.create("b"), root);
    c = new Artifact(PathFragment.create("c"), root);
    d = new Artifact(PathFragment.create("d"), root);
    CreateIncSymlinkAction action2 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(c, d, a, b), root.getPath());
    assertEquals(action1.computeKey(), action2.computeKey());
  }

  @Test
  public void testDifferentTargetsDifferentActionKey() throws Exception {
    Root root = Root.asDerivedRoot(rootDirectory, rootDirectory.getRelative("out"));
    Artifact a = new Artifact(PathFragment.create("a"), root);
    Artifact b = new Artifact(PathFragment.create("b"), root);
    CreateIncSymlinkAction action1 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b), root.getPath());
    // Can't reuse the artifacts here; that would lead to DuplicateArtifactException.
    a = new Artifact(PathFragment.create("a"), root);
    b = new Artifact(PathFragment.create("c"), root);
    CreateIncSymlinkAction action2 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b), root.getPath());
    assertThat(action2.computeKey()).isNotEqualTo(action1.computeKey());
  }

  @Test
  public void testDifferentSymlinksDifferentActionKey() throws Exception {
    Root root = Root.asDerivedRoot(rootDirectory, rootDirectory.getRelative("out"));
    Artifact a = new Artifact(PathFragment.create("a"), root);
    Artifact b = new Artifact(PathFragment.create("b"), root);
    CreateIncSymlinkAction action1 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b), root.getPath());
    // Can't reuse the artifacts here; that would lead to DuplicateArtifactException.
    a = new Artifact(PathFragment.create("c"), root);
    b = new Artifact(PathFragment.create("b"), root);
    CreateIncSymlinkAction action2 = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b), root.getPath());
    assertThat(action2.computeKey()).isNotEqualTo(action1.computeKey());
  }

  @Test
  public void testExecute() throws Exception {
    Path outputDir = rootDirectory.getRelative("out");
    outputDir.createDirectory();
    Root root = Root.asDerivedRoot(rootDirectory, outputDir);
    Path symlink = rootDirectory.getRelative("out/a");
    Artifact a = new Artifact(symlink, root);
    Artifact b = new Artifact(PathFragment.create("b"), root);
    CreateIncSymlinkAction action = new CreateIncSymlinkAction(NULL_ACTION_OWNER,
        ImmutableMap.of(a, b), outputDir);
    action.execute(null);
    symlink.stat(Symlinks.NOFOLLOW);
    assertThat(symlink.isSymbolicLink()).isTrue();
    assertEquals(symlink.readSymbolicLink(), b.getPath().asFragment());
    assertThat(rootDirectory.getRelative("a").exists()).isFalse();
  }

  @Test
  public void testFileRemoved() throws Exception {
    Path outputDir = rootDirectory.getRelative("out");
    outputDir.createDirectory();
    Root root = Root.asDerivedRoot(rootDirectory, outputDir);
    Path symlink = rootDirectory.getRelative("out/subdir/a");
    Artifact a = new Artifact(symlink, root);
    Artifact b = new Artifact(PathFragment.create("b"), root);
    CreateIncSymlinkAction action =
        new CreateIncSymlinkAction(NULL_ACTION_OWNER, ImmutableMap.of(a, b), outputDir);
    Path extra = rootDirectory.getRelative("out/extra");
    extra.getOutputStream().close();
    assertThat(extra.exists()).isTrue();
    action.prepare(rootDirectory);
    assertThat(extra.exists()).isFalse();
  }
}
