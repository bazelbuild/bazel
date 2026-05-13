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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ResourceSet.ResourceSetConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ResourceSet}. */
@RunWith(JUnit4.class)
public class ResourceSetTest {

  private ResourceSetConverter converter;

  @Before
  public final void createConverter() throws Exception  {
    converter = new ResourceSetConverter();
  }

  @Test
  public void testConverterParsesExpectedFormat() throws Exception {
    ResourceSet resources = converter.convert("1,0.5,2");
    assertThat(resources.getMemoryMb()).isWithin(0.01).of(1.0);
    assertThat(resources.getCpuUsage()).isWithin(0.01).of(0.5);
    assertThat(resources.getLocalTestCount()).isEqualTo(Integer.MAX_VALUE);
  }

  @Test
  public void testConverterThrowsWhenGivenInsufficientInputs() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("0,0,"));
  }

  @Test
  public void testConverterThrowsWhenGivenTooManyInputs() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("0,0,0,"));
  }

  @Test
  public void testConverterThrowsWhenGivenNegativeInputs() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("-1,0,0"));
  }

  @Test
  public void withResourceOverrides_noArgs_returnsSameInstance() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    assertThat(base.withResourceOverrides()).isSameInstanceAs(base);
  }

  @Test
  public void withResourceOverrides_allEmpty_returnsSameInstance() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    assertThat(base.withResourceOverrides(ImmutableMap.of(), ImmutableMap.of()))
        .isSameInstanceAs(base);
  }

  @Test
  public void withResourceOverrides_overridesExistingResource() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    ResourceSet result = base.withResourceOverrides(ImmutableMap.of("cpu", 4.0));
    assertThat(result.getCpuUsage()).isEqualTo(4.0);
    assertThat(result.getMemoryMb()).isEqualTo(100.0);
  }

  @Test
  public void withResourceOverrides_addsNewResource() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    ResourceSet result = base.withResourceOverrides(ImmutableMap.of("gpu", 2.0));
    assertThat(result.get("gpu")).isEqualTo(2.0);
    assertThat(result.getCpuUsage()).isEqualTo(1.0);
    assertThat(result.getMemoryMb()).isEqualTo(100.0);
  }

  @Test
  public void withResourceOverrides_laterOverrideWins() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    ResourceSet result =
        base.withResourceOverrides(ImmutableMap.of("cpu", 2.0), ImmutableMap.of("cpu", 8.0));
    assertThat(result.getCpuUsage()).isEqualTo(8.0);
  }

  @Test
  public void withResourceOverrides_mergesAcrossOverrides() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    ResourceSet result =
        base.withResourceOverrides(
            ImmutableMap.of("cpu", 4.0, "gpu", 1.0), ImmutableMap.of("memory", 2000.0));
    assertThat(result.getCpuUsage()).isEqualTo(4.0);
    assertThat(result.getMemoryMb()).isEqualTo(2000.0);
    assertThat(result.get("gpu")).isEqualTo(1.0);
  }

  @Test
  public void withResourceOverrides_preservesLocalTestCount() {
    ResourceSet base = ResourceSet.create(100, 1, 5);
    ResourceSet result = base.withResourceOverrides(ImmutableMap.of("cpu", 4.0));
    assertThat(result.getLocalTestCount()).isEqualTo(5);
  }

  @Test
  public void withResourceOverrides_skipsEmptyAmongNonEmpty() {
    ResourceSet base = ResourceSet.createWithRamCpu(100, 1);
    ResourceSet result =
        base.withResourceOverrides(
            ImmutableMap.of(), ImmutableMap.of("cpu", 4.0), ImmutableMap.of());
    assertThat(result.getCpuUsage()).isEqualTo(4.0);
  }
}
