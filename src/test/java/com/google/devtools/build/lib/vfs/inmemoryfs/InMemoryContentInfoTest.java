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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class InMemoryContentInfoTest {

  private Clock clock;

  @Before
  public final void createClock() throws Exception  {
    clock = BlazeClock.instance();
  }

  @Test
  public void testDirectoryCannotAddNullChild() {
    InMemoryDirectoryInfo directory = new InMemoryDirectoryInfo(clock);

    assertThrows(NullPointerException.class, () -> directory.addChild("bar", null));
  }

  @Test
  public void testDirectoryCannotAddChildTwice() {
    InMemoryDirectoryInfo directory = new InMemoryDirectoryInfo(clock);
    InMemoryFileInfo otherFile = new InMemoryFileInfo(clock);
    directory.addChild("bar", otherFile);

    assertThrows(IllegalArgumentException.class, () -> directory.addChild("bar", otherFile));
  }

  @Test
  public void testDirectoryRemoveNonExistingChild() {
    InMemoryDirectoryInfo directory = new InMemoryDirectoryInfo(clock);
    assertThrows(IllegalArgumentException.class, () -> directory.removeChild("bar"));
  }

}
