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

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.ObjectCodecTester;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RootTest}. */
@RunWith(JUnit4.class)
public class RootTest {
  private FileSystem fs;

  @Before
  public final void initializeFileSystem() throws Exception {
    fs = new InMemoryFileSystem(BlazeClock.instance());
  }

  @Test
  public void testEqualsAndHashCodeContract() throws Exception {
    new EqualsTester()
        .addEqualityGroup(Root.fromFileSystemRoot(fs), Root.fromFileSystemRoot(fs))
        .addEqualityGroup(Root.fromPath(fs.getPath("/foo")), Root.fromPath(fs.getPath("/foo")))
        .testEquals();
  }

  @Test
  public void testPathRoot() throws Exception {
    Root root = Root.fromPath(fs.getPath("/foo"));
    assertThat(root.asPath()).isEqualTo(fs.getPath("/foo"));
    assertThat(root.contains(fs.getPath("/foo/bar"))).isTrue();
    assertThat(root.contains(fs.getPath("/boo/bar"))).isFalse();
    assertThat(root.getRelative(PathFragment.create("bar"))).isEqualTo(fs.getPath("/foo/bar"));
    assertThat(root.getRelative("bar")).isEqualTo(fs.getPath("/foo/bar"));
    assertThat(root.relativize(fs.getPath("/foo/bar"))).isEqualTo(PathFragment.create("bar"));
  }

  @Test
  public void testFileSystemRootPath() throws Exception {
    Root root = Root.fromFileSystemRoot(fs);
    assertThat(root.asPath()).isEqualTo(fs.getPath("/"));
    assertThat(root.contains(fs.getPath("/foo"))).isTrue();
    assertThat(root.getRelative(PathFragment.create("foo"))).isEqualTo(fs.getPath("/foo"));
    assertThat(root.getRelative("foo")).isEqualTo(fs.getPath("/foo"));
    assertThat(root.relativize(fs.getPath("/foo"))).isEqualTo(PathFragment.create("foo"));
  }

  @Test
  public void testSerialization() throws Exception {
    ObjectCodec<Root> codec = Root.getCodec(new PathCodec(fs));
    ObjectCodecTester.newBuilder(codec)
        .verificationFunction((a, b) -> a.equals(b))
        .addSubjects(Root.fromFileSystemRoot(fs), Root.fromPath(fs.getPath("/foo")))
        .buildAndRunTests();
  }
}
