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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for {@link ActionInputMap}. */
@RunWith(JUnit4.class)
public final class ActionInputMapTest {

  private ActionInputMap map;

  @Before
  public void init() {
    map = new ActionInputMap(1); // small hint to stress the map
  }

  @Test
  public void basicPutAndLookup() {
    assertThat(put("/abc/def", 5)).isTrue();
    assertThat(map.size()).isEqualTo(1);
    assertContains("/abc/def", 5);
    assertThat(map.getMetadata("blah")).isNull();
    assertThat(map.getInput("blah")).isNull();
  }

  @Test
  public void ignoresSubsequent() {
    assertThat(put("/abc/def", 5)).isTrue();
    assertThat(map.size()).isEqualTo(1);
    assertThat(put("/abc/def", 6)).isFalse();
    assertThat(map.size()).isEqualTo(1);
    assertThat(put("/ghi/jkl", 7)).isTrue();
    assertThat(map.size()).isEqualTo(2);
    assertThat(put("/ghi/jkl", 8)).isFalse();
    assertThat(map.size()).isEqualTo(2);
    assertContains("/abc/def", 5);
    assertContains("/ghi/jkl", 7);
  }

  @Test
  public void clear() {
    assertThat(put("/abc/def", 5)).isTrue();
    assertThat(map.size()).isEqualTo(1);
    assertThat(put("/ghi/jkl", 7)).isTrue();
    assertThat(map.size()).isEqualTo(2);
    map.clear();
    assertThat(map.size()).isEqualTo(0);
    assertThat(map.getMetadata("/abc/def")).isNull();
    assertThat(map.getMetadata("/ghi/jkl")).isNull();
  }

  @Test
  public void stress() {
    ArrayList<TestEntry> data = new ArrayList<>();
    {
      Random rng = new Random();
      HashSet<TestInput> deduper = new HashSet<>();
      for (int i = 0; i < 100000; ++i) {
        byte[] bytes = new byte[80];
        rng.nextBytes(bytes);
        for (int j = 0; j < bytes.length; ++j) {
          bytes[j] &= ((byte) 0x7f);
        }
        TestInput nextInput = new TestInput(new String(bytes, US_ASCII));
        if (deduper.add(nextInput)) {
          data.add(new TestEntry(nextInput, new TestMetadata(i)));
        }
      }
    }
    for (int iteration = 0; iteration < 20; ++iteration) {
      map.clear();
      Collections.shuffle(data);
      for (int i = 0; i < data.size(); ++i) {
        TestEntry entry = data.get(i);
        assertThat(map.putWithNoDepOwner(entry.input, entry.metadata)).isTrue();
      }
      assertThat(map.size()).isEqualTo(data.size());
      for (int i = 0; i < data.size(); ++i) {
        TestEntry entry = data.get(i);
        assertThat(map.getMetadata(entry.input)).isEqualTo(entry.metadata);
      }
    }
  }

  private boolean put(String execPath, int value) {
    return map.putWithNoDepOwner(new TestInput(execPath), new TestMetadata(value));
  }

  private void assertContains(String execPath, int value) {
    assertThat(map.getMetadata(new TestInput(execPath))).isEqualTo(new TestMetadata(value));
    assertThat(map.getMetadata(execPath)).isEqualTo(new TestMetadata(value));
    assertThat(map.getInput(execPath)).isEqualTo(new TestInput(execPath));
  }

  private static class TestEntry {
    public final TestInput input;
    public final TestMetadata metadata;

    public TestEntry(TestInput input, TestMetadata metadata) {
      this.input = input;
      this.metadata = metadata;
    }
  }

  private static class TestInput implements ActionInput {
    private final PathFragment fragment;

    public TestInput(String fragment) {
      this.fragment = PathFragment.create(fragment);
    }

    @Override
    public PathFragment getExecPath() {
      return fragment;
    }

    @Override
    public String getExecPathString() {
      return fragment.toString();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof TestInput)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      return fragment.equals(((TestInput) other).fragment);
    }

    @Override
    public int hashCode() {
      return fragment.hashCode();
    }
  }

  private static class TestMetadata extends FileArtifactValue {
    private final int id;

    public TestMetadata(int id) {
      this.id = id;
    }

    @Override
    public FileStateType getType() {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte[] getDigest() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      throw new UnsupportedOperationException();
    }

    @Override
    @SuppressWarnings("EqualsHashCode")
    public boolean equals(Object o) {
      if (!(o instanceof TestMetadata)) {
        return false;
      }
      return id == ((TestMetadata) o).id;
    }
  }
}
