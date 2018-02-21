// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.events;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class LocationTest extends EventTestTemplate {

  @Test
  public void fromFile() throws Exception {
    Location location = Location.fromPathFragment(path);
    assertThat(location.getPath()).isEqualTo(path);
    assertThat(location.getStartOffset()).isEqualTo(0);
    assertThat(location.getEndOffset()).isEqualTo(0);
    assertThat(location.getStartLineAndColumn()).isNull();
    assertThat(location.getEndLineAndColumn()).isNull();
    assertThat(location.print()).isEqualTo(path + ":1");
  }

  @Test
  public void testPrintRelative() throws Exception {
    Location location = Location.fromPathFragment(path);
    assertThat(location.print(PathFragment.create("/some/other/path"), PathFragment.create("baz")))
        .isEqualTo("/path/to/workspace/my/sample/path.txt:1");
    assertThat(
            location.print(
                PathFragment.create("/path/to/workspace/my"), PathFragment.create("new")))
        .isEqualTo("new/sample/path.txt:1");
    assertThat(
            location.print(
                PathFragment.create("/path/to/workspace/my/sample"), PathFragment.create("new")))
        .isEqualTo("new/path.txt:1");
    assertThat(
            location.print(
                PathFragment.create("/path/to/workspace/my/sample/path.txt"),
                PathFragment.create("new")))
        .isEqualTo("new:1");
  }

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            Location.fromPathFragment(path),
            Location.fromPathAndStartColumn(path, 0, 100, new Location.LineAndColumn(20, 25)),
            Location.BUILTIN)
        .runTests();
  }
}
