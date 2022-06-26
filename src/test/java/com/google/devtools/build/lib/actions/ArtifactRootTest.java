// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.protobuf.ByteString;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ArtifactRoot}. */
@RunWith(JUnit4.class)
public class ArtifactRootTest {
  private final Scratch scratch = new Scratch();

  @Test
  public void asSourceRoot_createsValidSourceRoot() throws IOException {
    Path sourceDir = scratch.dir("/source");
    ArtifactRoot root = ArtifactRoot.asSourceRoot(Root.fromPath(sourceDir));
    assertThat(root.isSourceRoot()).isTrue();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(sourceDir));
    assertThat(root.toString()).isEqualTo("/source[source]");
  }

  @Test
  public void asSourceRoot_nullRoot_fails() {
    assertThrows(NullPointerException.class, () -> ArtifactRoot.asSourceRoot(null));
  }

  @Test
  public void asDerivedRoot_createsValidDerivedRoot() throws IOException {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/root");

    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "root");

    assertThat(root.isSourceRoot()).isFalse();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.create("root"));
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(rootDir));
    assertThat(root.toString()).isEqualTo("/exec/root[derived]");
  }

  @Test
  public void asDerivedRoot_derivedRootIsExecRoot_failsNotOk() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, ""));
  }

  @Test
  public void asDerivedRoot_emptyPrefix_createsArtifactRoot() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThat(ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "", "suffix", ""))
        .isEqualTo(ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "suffix"));
  }

  @Test
  public void asDerivedRoot_prefixWithSlash_fails() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "suffix/"));
  }

  @Test
  public void asDerivedRoot_noPrefixes_fails() throws IOException {
    Path execRoot = scratch.dir("/exec");
    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output));
  }

  @Test
  public void asDerivedRoot_nullExecPath_fails() {
    assertThrows(
        NullPointerException.class,
        () -> ArtifactRoot.asDerivedRoot(null, RootType.Output, "exec"));
  }

  @Test
  public void asDerivedRootPathFragment_simpleExecPath_createsArtifactRoot() throws Exception {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/root");

    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, PathFragment.create("root"));

    assertThat(root.isSourceRoot()).isFalse();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.create("root"));
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(rootDir));
    assertThat(root.toString()).isEqualTo("/exec/root[derived]");
  }

  @Test
  public void asDerivedRootPathFragment_nestedExecPath_createsArtifactRoot() throws Exception {
    Path execRoot = scratch.dir("/exec");
    Path rootDir = scratch.dir("/exec/dir1/dir2/dir3");

    ArtifactRoot root =
        ArtifactRoot.asDerivedRoot(
            execRoot, RootType.Output, PathFragment.create("dir1/dir2/dir3"));

    assertThat(root.isSourceRoot()).isFalse();
    assertThat(root.getExecPath()).isEqualTo(PathFragment.create("dir1/dir2/dir3"));
    assertThat(root.getRoot()).isEqualTo(Root.fromPath(rootDir));
    assertThat(root.toString()).isEqualTo("/exec/dir1/dir2/dir3[derived]");
  }

  @Test
  public void asDerivedRootPathFragment_emptyExecPath_fails() throws Exception {
    Path execRoot = scratch.dir("/exec");

    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, PathFragment.EMPTY_FRAGMENT));
  }

  @Test
  public void asDerivedRootPathFragment_execPathIsCurrentDirectory_fails() throws Exception {
    Path execRoot = scratch.dir("/exec");

    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, PathFragment.create(".")));
  }

  @Test
  public void asDerivedRootPathFragment_execPathIsDirectoryUp_fails() throws Exception {
    Path execRoot = scratch.dir("/exec");

    assertThrows(
        IllegalArgumentException.class,
        () -> ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, PathFragment.create("..")));
  }

  @Test
  public void asDerivedRootPathFragment_execPathContainsDirectoryUp_fails() throws Exception {
    Path execRoot = scratch.dir("/exec");

    assertThrows(
        IllegalArgumentException.class,
        () ->
            ArtifactRoot.asDerivedRoot(
                execRoot, RootType.Output, PathFragment.create("../outsideExecRoot")));
  }

  @Test
  public void derivedRootSerialization_rootMatchesDesignatedLikelyRoot_skipsRootInSerialization()
      throws Exception {
    Path execRoot = scratch.dir("/thisisaveryverylongexecrootthatwedontwanttoserialize");
    ArtifactRoot derivedRoot =
        ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "first", "second", "third");
    ObjectCodecRegistry registry = AutoRegistry.get();
    ImmutableClassToInstanceMap<Object> dependencies =
        ImmutableClassToInstanceMap.builder()
            .put(FileSystem.class, scratch.getFileSystem())
            .put(
                Root.RootCodecDependencies.class,
                new Root.RootCodecDependencies(/*likelyPopularRoot=*/ Root.fromPath(execRoot)))
            .build();
    ObjectCodecRegistry.Builder registryBuilder = registry.getBuilder();
    for (Object val : dependencies.values()) {
      registryBuilder.addReferenceConstant(val);
    }
    ObjectCodecs objectCodecs = new ObjectCodecs(registryBuilder.build(), dependencies);
    ByteString serialized = objectCodecs.serialize(derivedRoot);
    // 30 bytes as of 2020/04/27.
    assertThat(serialized.size()).isLessThan(31);
  }

  @Test
  public void equals_returnsTrueForIdenticalRootAndDetectsDifferencesOnEachField()
      throws IOException {
    Path execRoot = scratch.dir("/exec");
    String rootSegment = "root";
    Path rootDir = execRoot.getChild(rootSegment);
    rootDir.createDirectoryAndParents();
    Path otherRootDir = scratch.dir("/");
    Path sourceDir = scratch.dir("/source");

    new EqualsTester()
        .addEqualityGroup(
            ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, rootSegment),
            ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, PathFragment.create(rootSegment)))
        .addEqualityGroup(
            ArtifactRoot.asDerivedRoot(otherRootDir, RootType.Output, "exec", rootSegment))
        .addEqualityGroup(ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "otherSegment"))
        .addEqualityGroup(ArtifactRoot.asSourceRoot(Root.fromPath(sourceDir)))
        .addEqualityGroup(ArtifactRoot.asSourceRoot(Root.fromPath(rootDir)))
        .testEquals();
  }
}
