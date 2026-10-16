// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionRequirements}. */
@RunWith(JUnit4.class)
public class ExecutionRequirementsTest {

  @Test
  public void parseResources_empty() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of())).isEmpty();
  }

  @Test
  public void parseResources_ignoresUnrelatedKeys() throws Exception {
    assertThat(
            ExecutionRequirements.parseResources(
                ImmutableMap.of("pool", "default", "no-sandbox", "", "local", "")))
        .isEmpty();
  }

  // exec_properties format: key = "resources:name", value = "amount"

  @Test
  public void parseResources_execProp_cpu() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu", "4")))
        .containsExactly("cpu", 4.0);
  }

  @Test
  public void parseResources_execProp_memory() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:memory", "2000")))
        .containsExactly("memory", 2000.0);
  }

  @Test
  public void parseResources_execProp_multiple() throws Exception {
    assertThat(
            ExecutionRequirements.parseResources(
                ImmutableMap.of("resources:cpu", "8", "resources:memory", "4000")))
        .containsExactly("cpu", 8.0, "memory", 4000.0);
  }

  @Test
  public void parseResources_execProp_customResource() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:gpu", "2")))
        .containsExactly("gpu", 2.0);
  }

  @Test
  public void parseResources_execProp_floatingPoint() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu", "2.5")))
        .containsExactly("cpu", 2.5);
  }

  @Test
  public void parseResources_execProp_mixedKeys() throws Exception {
    assertThat(
            ExecutionRequirements.parseResources(
                ImmutableMap.of(
                    "pool", "default",
                    "resources:cpu", "4",
                    "container-image", "docker://foo",
                    "resources:gpu", "1")))
        .containsExactly("cpu", 4.0, "gpu", 1.0);
  }

  // Tag format: key = "resources:name:amount" or "cpu:amount", value = ""

  @Test
  public void parseResources_tag_resources() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu:4", "")))
        .containsExactly("cpu", 4.0);
  }

  @Test
  public void parseResources_tag_cpu() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("cpu:2", "")))
        .containsExactly("cpu", 2.0);
  }

  @Test
  public void parseResources_tag_customResource() throws Exception {
    assertThat(ExecutionRequirements.parseResources(ImmutableMap.of("resources:gpu:3", "")))
        .containsExactly("gpu", 3.0);
  }

  @Test
  public void parseResources_tag_multiple() throws Exception {
    assertThat(
            ExecutionRequirements.parseResources(
                ImmutableMap.of("resources:gpu:2", "", "cpu:4", "")))
        .containsExactly("gpu", 2.0, "cpu", 4.0);
  }

  // Validation

  @Test
  public void parseResources_throwsOnTagMissingAmount() {
    assertThrows(
        UserExecException.class,
        () -> ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu:", "")));
  }

  @Test
  public void parseResources_throwsOnExecPropMissingAmount() {
    assertThrows(
        UserExecException.class,
        () -> ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu", "")));
  }

  @Test
  public void parseResources_throwsOnInvalidTagValue() {
    assertThrows(
        UserExecException.class,
        () ->
            ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu:notanumber", "")));
  }

  @Test
  public void parseResources_throwsOnInvalidExecPropValue() {
    assertThrows(
        UserExecException.class,
        () -> ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu", "notanumber")));
  }

  @Test
  public void parseResources_throwsOnNegativeTagValue() {
    assertThrows(
        UserExecException.class,
        () -> ExecutionRequirements.parseResources(ImmutableMap.of("resources:cpu:-1", "")));
  }

  @Test
  public void parseResources_throwsOnDuplicateResource() {
    assertThrows(
        UserExecException.class,
        () ->
            ExecutionRequirements.parseResources(
                ImmutableMap.of("resources:cpu:4", "", "cpu:2", "")));
  }
}
