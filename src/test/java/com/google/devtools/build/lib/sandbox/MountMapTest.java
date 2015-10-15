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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

/**
 * Tests for {@code MountMap}.
 */
@RunWith(JUnit4.class)
public class MountMapTest extends LinuxSandboxedStrategyTestCase {
  @Test
  public void testMountMapWithNormalMounts() throws IOException {
    // Allowed: Just two normal mounts (a -> sandbox/a, b -> sandbox/b)
    MountMap mounts = new MountMap();
    mounts.put(fileSystem.getPath("/a"), workspaceDir.getRelative("a"));
    mounts.put(fileSystem.getPath("/b"), workspaceDir.getRelative("b"));
    assertThat(mounts)
        .isEqualTo(
            ImmutableMap.of(
                fileSystem.getPath("/a"), workspaceDir.getRelative("a"),
                fileSystem.getPath("/b"), workspaceDir.getRelative("b")));
  }

  @Test
  public void testMountMapWithSameMountTwice() throws IOException {
    // Allowed: Mount same thing twice (a -> sandbox/a, a -> sandbox/a, b -> sandbox/b)
    MountMap mounts = new MountMap();
    mounts.put(fileSystem.getPath("/a"), workspaceDir.getRelative("a"));
    mounts.put(fileSystem.getPath("/a"), workspaceDir.getRelative("a"));
    mounts.put(fileSystem.getPath("/b"), workspaceDir.getRelative("b"));
    assertThat(mounts)
        .isEqualTo(
            ImmutableMap.of(
                fileSystem.getPath("/a"), workspaceDir.getRelative("a"),
                fileSystem.getPath("/b"), workspaceDir.getRelative("b")));
  }

  @Test
  public void testMountMapWithOneThingTwoTargets() throws IOException {
    // Allowed: Mount one thing in two targets (x -> sandbox/a, x -> sandbox/b)
    MountMap mounts = new MountMap();
    mounts.put(fileSystem.getPath("/a"), workspaceDir.getRelative("x"));
    mounts.put(fileSystem.getPath("/b"), workspaceDir.getRelative("x"));
    assertThat(mounts)
        .isEqualTo(
            ImmutableMap.of(
                fileSystem.getPath("/a"), workspaceDir.getRelative("x"),
                fileSystem.getPath("/b"), workspaceDir.getRelative("x")));
  }

  @Test
  public void testMountMapWithTwoThingsOneTarget() throws IOException {
    // Forbidden: Mount two things onto the same target (x -> sandbox/a, y -> sandbox/a)
    try {
      MountMap mounts = new MountMap();
      mounts.put(fileSystem.getPath("/x"), workspaceDir.getRelative("a"));
      mounts.put(fileSystem.getPath("/x"), workspaceDir.getRelative("b"));
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessage(
              String.format(
                  "Cannot mount both '%s' and '%s' onto '%s'",
                  workspaceDir.getRelative("a"),
                  workspaceDir.getRelative("b"),
                  fileSystem.getPath("/x")));
    }
  }

  @Test
  public void testMountMapGuaranteesOrdering() {
    MountMap mounts = new MountMap();
    mounts.put(fileSystem.getPath("/a/c"), workspaceDir.getRelative("x"));
    mounts.put(fileSystem.getPath("/b"), workspaceDir.getRelative("x"));
    mounts.put(fileSystem.getPath("/a/b"), workspaceDir.getRelative("x"));
    mounts.put(fileSystem.getPath("/a"), workspaceDir.getRelative("x"));

    assertThat(mounts.entrySet())
        .containsExactlyElementsIn(
            ImmutableMap.builder()
                .put(fileSystem.getPath("/a"), workspaceDir.getRelative("x"))
                .put(fileSystem.getPath("/a/b"), workspaceDir.getRelative("x"))
                .put(fileSystem.getPath("/a/c"), workspaceDir.getRelative("x"))
                .put(fileSystem.getPath("/b"), workspaceDir.getRelative("x"))
                .build()
                .entrySet())
        .inOrder();
  }
}
