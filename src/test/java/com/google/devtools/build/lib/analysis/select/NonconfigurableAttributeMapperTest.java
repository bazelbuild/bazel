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
package com.google.devtools.build.lib.analysis.select;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link NonconfigurableAttributeMapper}.
 */
@RunWith(JUnit4.class)
public class NonconfigurableAttributeMapperTest extends AbstractAttributeMapperTest {

  private Rule rule;

  @Before
  public final void createRule() throws Exception {
    rule = scratchRule("x", "myrule",
        "cc_binary(",
        "    name = 'myrule',",
        "    srcs = ['a', 'b', 'c'],",
        "    linkstatic = 1,",
        "  deprecation = \"this rule is deprecated!\")");
  }

  @Test
  public void testGetNonconfigurableAttribute() throws Exception {
    assertThat(NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING))
        .isEqualTo("this rule is deprecated!");
  }

  @Test
  public void testGetConfigurableAttribute() throws Exception {
    IllegalStateException e =
        assertThrows(
            "Expected NonconfigurableAttributeMapper to fail on a configurable attribute type",
            IllegalStateException.class,
            () -> NonconfigurableAttributeMapper.of(rule).get("linkstatic", Type.BOOLEAN));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Attribute 'linkstatic' is potentially configurable - not allowed here");
  }

  @Test
  public void testGet_nonexistentAttribute() throws Exception {
    IllegalArgumentException e =
        assertThrows(
            "Expected NonconfigurableAttributeMapper to fail on nonexistent attribute name",
            IllegalArgumentException.class,
            () -> NonconfigurableAttributeMapper.of(rule).get("nonexistent-attr", Type.STRING));
    assertThat(e).hasMessageThat().contains("No such attribute nonexistent-attr in cc_binary");
  }
}
