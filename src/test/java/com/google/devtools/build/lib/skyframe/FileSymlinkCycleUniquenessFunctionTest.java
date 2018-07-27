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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FileSymlinkCycleUniquenessFunctionTest {

  @Test
  public void testHashCodeAndEqualsContract() throws Exception {
    Root root = Root.fromPath(new InMemoryFileSystem().getPath("/root"));
    RootedPath p1 = RootedPath.toRootedPath(root, PathFragment.create("p1"));
    RootedPath p2 = RootedPath.toRootedPath(root, PathFragment.create("p2"));
    RootedPath p3 = RootedPath.toRootedPath(root, PathFragment.create("p3"));
    ImmutableList<RootedPath> cycleA1 = ImmutableList.of(p1);
    ImmutableList<RootedPath> cycleB1 = ImmutableList.of(p2);
    ImmutableList<RootedPath> cycleC1 = ImmutableList.of(p1, p2, p3);
    ImmutableList<RootedPath> cycleC2 = ImmutableList.of(p2, p3, p1);
    ImmutableList<RootedPath> cycleC3 = ImmutableList.of(p3, p1, p2);
    new EqualsTester()
        .addEqualityGroup(FileSymlinkCycleUniquenessFunction.key(cycleA1))
        .addEqualityGroup(FileSymlinkCycleUniquenessFunction.key(cycleB1))
        .addEqualityGroup(
            FileSymlinkCycleUniquenessFunction.key(cycleC1),
            FileSymlinkCycleUniquenessFunction.key(cycleC2),
            FileSymlinkCycleUniquenessFunction.key(cycleC3))
        .testEquals();
  }
}
