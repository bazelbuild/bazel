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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class LocationTest extends EventTestTemplate {

  @Test
  public void fromFile() throws Exception {
    Location location = Location.fromPathFragment(path);
    assertEquals(path, location.getPath());
    assertEquals(0, location.getStartOffset());
    assertEquals(0, location.getEndOffset());
    assertNull(location.getStartLineAndColumn());
    assertNull(location.getEndLineAndColumn());
    assertEquals(path + ":1", location.print());
  }
  
  @Test
  public void testPrintRelative() throws Exception {
    Location location = Location.fromPathFragment(path);
    assertEquals(
        "/path/to/workspace/my/sample/path.txt:1",
        location.print(PathFragment.create("/some/other/path"), PathFragment.create("baz")));
    assertEquals(
        "new/sample/path.txt:1",
        location.print(PathFragment.create("/path/to/workspace/my"), PathFragment.create("new")));
    assertEquals(
        "new/path.txt:1",
        location.print(
            PathFragment.create("/path/to/workspace/my/sample"), PathFragment.create("new")));
    assertEquals(
        "new:1",
        location.print(
            PathFragment.create("/path/to/workspace/my/sample/path.txt"),
            PathFragment.create("new")));
  }
}
