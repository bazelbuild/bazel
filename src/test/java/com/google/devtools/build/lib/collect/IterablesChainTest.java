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
package com.google.devtools.build.lib.collect;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;

/**
 * A test for {@link IterablesChain}.
 */
@RunWith(JUnit4.class)
public class IterablesChainTest {

  @Test
  public void addElement() {
    IterablesChain.Builder<String> builder = IterablesChain.builder();
    builder.addElement("a");
    builder.addElement("b");
    assertEquals(Arrays.asList("a", "b"), ImmutableList.copyOf(builder.build()));
  }

  @Test
  public void add() {
    IterablesChain.Builder<String> builder = IterablesChain.builder();
    builder.add(ImmutableList.of("a", "b"));
    assertEquals(Arrays.asList("a", "b"), ImmutableList.copyOf(builder.build()));
  }

  @Test
  public void isEmpty() {
    IterablesChain.Builder<String> builder = IterablesChain.builder();
    assertTrue(builder.isEmpty());
    builder.addElement("a");
    assertFalse(builder.isEmpty());
    builder = IterablesChain.builder();
    assertTrue(builder.isEmpty());
    builder.add(ImmutableList.of("a"));
    assertFalse(builder.isEmpty());
  }
}
