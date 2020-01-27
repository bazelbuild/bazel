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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.actions.ResourceSet.ResourceSetConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for @{link ResourceSet}.
 */
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

  @Test(expected = OptionsParsingException.class)
  public void testConverterThrowsWhenGivenInsufficientInputs() throws Exception {
    converter.convert("0,0,");
    fail();
  }

  @Test(expected = OptionsParsingException.class)
  public void testConverterThrowsWhenGivenTooManyInputs() throws Exception {
    converter.convert("0,0,0,");
    fail();
  }

  @Test(expected = OptionsParsingException.class)
  public void testConverterThrowsWhenGivenNegativeInputs() throws Exception {
    converter.convert("-1,0,0");
    fail();
  }
}
