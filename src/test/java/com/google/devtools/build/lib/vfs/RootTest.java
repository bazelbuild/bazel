// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Comparator;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RootTest}. */
@RunWith(JUnit4.class)
public class RootTest {
  private FileSystem fs;

  @Before
  public final void initializeFileSystem() {
    fs = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
  }

  @Test
  public void testEqualsAndHashCodeContract() {
    FileSystem otherFs = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    new EqualsTester()
        .addEqualityGroup(Root.absoluteRoot(fs), Root.absoluteRoot(fs))
        .addEqualityGroup(Root.absoluteRoot(otherFs), Root.absoluteRoot(otherFs))
        .addEqualityGroup(Root.fromPath(fs.getPath("/foo")), Root.fromPath(fs.getPath("/foo")))
        .testEquals();
  }

  @Test
  public void testPathRoot() {
    Root root = Root.fromPath(fs.getPath("/foo"));
    assertThat(root.asPath()).isEqualTo(fs.getPath("/foo"));
    assertThat(root.contains(fs.getPath("/foo/bar"))).isTrue();
    assertThat(root.contains(fs.getPath("/boo/bar"))).isFalse();
    assertThat(root.contains(PathFragment.create("/foo/bar"))).isTrue();
    assertThat(root.contains(PathFragment.create("foo/bar"))).isFalse();
    assertThat(root.getRelative(PathFragment.create("bar"))).isEqualTo(fs.getPath("/foo/bar"));
    assertThat(root.getRelative("bar")).isEqualTo(fs.getPath("/foo/bar"));
    assertThat(root.getRelative(PathFragment.create("/bar"))).isEqualTo(fs.getPath("/bar"));
    assertThat(root.relativize(fs.getPath("/foo/bar"))).isEqualTo(PathFragment.create("bar"));
    assertThat(root.relativize(PathFragment.create("/foo/bar")))
        .isEqualTo(PathFragment.create("bar"));
    assertThrows(IllegalArgumentException.class, () -> root.relativize(PathFragment.create("foo")));
  }

  @Test
  public void testFilesystemTransform() {
    FileSystem fs2 = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    Root root = Root.fromPath(fs.getPath("/foo"));
    Root root2 = Root.toFileSystem(root, fs2);
    assertThat(root2.asPath().getFileSystem()).isSameInstanceAs(fs2);
    assertThat(root2.asPath().asFragment()).isEqualTo(PathFragment.create("/foo"));
    assertThat(root.isAbsolute()).isFalse();
  }

  @Test
  public void testFileSystemAbsoluteRoot() {
    Root root = Root.absoluteRoot(fs);
    assertThat(root.asPath()).isNull();
    assertThat(root.contains(fs.getPath("/foo"))).isTrue();
    assertThat(root.contains(PathFragment.create("/foo/bar"))).isTrue();
    assertThat(root.contains(PathFragment.create("foo/bar"))).isFalse();
    assertThat(root.getRelative("/foo")).isEqualTo(fs.getPath("/foo"));
    assertThat(root.relativize(fs.getPath("/foo"))).isEqualTo(PathFragment.create("/foo"));
    assertThat(root.relativize(PathFragment.create("/foo"))).isEqualTo(PathFragment.create("/foo"));

    assertThrows(
        IllegalArgumentException.class, () -> root.getRelative(PathFragment.create("foo")));
    assertThrows(
        IllegalArgumentException.class, () -> root.getRelative(PathFragment.create("foo")));
    assertThrows(IllegalArgumentException.class, () -> root.relativize(PathFragment.create("foo")));
  }

  @Test
  public void testCompareTo() {
    Root a = Root.fromPath(fs.getPath("/a"));
    Root b = Root.fromPath(fs.getPath("/b"));
    Root root = Root.absoluteRoot(fs);
    List<Root> list = Lists.newArrayList(a, root, b);
    list.sort(Comparator.naturalOrder());
    assertThat(list).containsExactly(root, a, b).inOrder();
  }

  @Test
  public void testSerialization_simple() throws Exception {
    Root fooPathRoot = Root.fromPath(fs.getPath("/foo"));
    Root barPathRoot = Root.fromPath(fs.getPath("/bar"));
    new SerializationTester(Root.absoluteRoot(fs), fooPathRoot, barPathRoot)
        .addDependency(FileSystem.class, fs)
        .addDependency(
            Root.RootCodecDependencies.class,
            new Root.RootCodecDependencies(/*likelyPopularRoot=*/ fooPathRoot))
        .runTests();
  }

  @Test
  public void testSerialization_likelyPopularRootIsCanonicalized() throws Exception {
    Root fooPathRoot = Root.fromPath(fs.getPath("/foo"));
    Root otherFooPathRoot = Root.fromPath(fs.getPath("/foo"));
    Root barPathRoot = Root.fromPath(fs.getPath("/bar"));
    Root bazPathRoot = Root.fromPath(fs.getPath("/baz"));
    Root fsAabsoluteRoot = Root.absoluteRoot(fs);

    assertThat(fooPathRoot).isNotSameInstanceAs(otherFooPathRoot);
    assertThat(fooPathRoot).isEqualTo(otherFooPathRoot);

    ObjectCodecRegistry registry = AutoRegistry.get();
    ImmutableClassToInstanceMap<Object> dependencies =
        ImmutableClassToInstanceMap.builder()
            .put(FileSystem.class, fs)
            .put(
                Root.RootCodecDependencies.class,
                new Root.RootCodecDependencies(
                    /*likelyPopularRoots=*/ ImmutableList.of(fooPathRoot, bazPathRoot)))
            .build();
    ObjectCodecRegistry.Builder registryBuilder = registry.getBuilder();
    for (Object val : dependencies.values()) {
      registryBuilder.addReferenceConstant(val);
    }
    ObjectCodecs objectCodecs = new ObjectCodecs(registryBuilder.build(), dependencies);

    Root fooPathRootDeserialized =
        (Root) objectCodecs.deserialize(objectCodecs.serialize(fooPathRoot));
    Root otherFooPathRootDeserialized =
        (Root) objectCodecs.deserialize(objectCodecs.serialize(otherFooPathRoot));
    assertThat(fooPathRootDeserialized).isSameInstanceAs(fooPathRoot);
    assertThat(otherFooPathRootDeserialized).isSameInstanceAs(fooPathRoot);

    Root barPathRootDeserialized =
        (Root) objectCodecs.deserialize(objectCodecs.serialize(barPathRoot));
    assertThat(barPathRootDeserialized).isNotSameInstanceAs(barPathRoot);
    assertThat(barPathRootDeserialized).isEqualTo(barPathRoot);

    Root bazPathRootDeserialized =
        (Root) objectCodecs.deserialize(objectCodecs.serialize(bazPathRoot));
    assertThat(bazPathRootDeserialized).isSameInstanceAs(bazPathRoot);

    Root fsAabsoluteRootDeserialized =
        (Root) objectCodecs.deserialize(objectCodecs.serialize(fsAabsoluteRoot));
    assertThat(fsAabsoluteRootDeserialized).isNotSameInstanceAs(fsAabsoluteRoot);
    assertThat(fsAabsoluteRootDeserialized).isEqualTo(fsAabsoluteRoot);
  }
}
