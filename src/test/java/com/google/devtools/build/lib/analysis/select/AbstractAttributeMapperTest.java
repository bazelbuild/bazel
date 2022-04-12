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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link AbstractAttributeMapper}. */
@RunWith(JUnit4.class)
public class AbstractAttributeMapperTest extends BuildViewTestCase {

  protected Rule rule;
  protected AbstractAttributeMapper mapper;

  private static final class TestMapper extends AbstractAttributeMapper {
    TestMapper(Rule rule) {
      super(rule);
    }
  }

  @Before
  public final void initializeRuleAndMapper() throws Exception {
    rule = scratchRule("p", "myrule",
        "cc_binary(name = 'myrule',",
        "          srcs = ['a', 'b', 'c'])");
    mapper = new TestMapper(rule);
  }

  @Test
  public void testRuleProperties() throws Exception {
    assertThat(mapper.getName()).isEqualTo(rule.getName());
    assertThat(mapper.getLabel()).isEqualTo(rule.getLabel());
  }

  @Test
  public void testPackageDefaultProperties() throws Exception {
    rule = scratchRule("a", "myrule",
        "cc_binary(name = 'myrule',",
        "          srcs = ['a', 'b', 'c'])");
    Package pkg = rule.getPackage();
    assertThat(mapper.getPackageDefaultHdrsCheck()).isEqualTo(pkg.getDefaultHdrsCheck());
    assertThat(mapper.getPackageDefaultTestOnly()).isEqualTo(pkg.getDefaultTestOnly());
    assertThat(mapper.getPackageDefaultDeprecation()).isEqualTo(pkg.getDefaultDeprecation());
  }

  @Test
  public void testAttributeTypeChecking() throws Exception {
    // Good typing:
    mapper.get("srcs", BuildType.LABEL_LIST);

    // Bad typing:
    assertThrows(
        "Expected type mismatch to trigger an exception",
        IllegalArgumentException.class,
        () -> mapper.get("srcs", Type.BOOLEAN));

    // Unknown attribute:
    assertThrows(
        "Expected type mismatch to trigger an exception",
        IllegalArgumentException.class,
        () -> mapper.get("nonsense", Type.BOOLEAN));
  }

  @Test
  public void testGetAttributeType() throws Exception {
    assertThat(mapper.getAttributeType("srcs")).isEqualTo(BuildType.LABEL_LIST);
    assertThat(mapper.getAttributeType("nonsense")).isNull();
  }

  @Test
  public void testGetAttributeDefinition() {
    assertThat(mapper.getAttributeDefinition("srcs").getName()).isEqualTo("srcs");
    assertThat(mapper.getAttributeDefinition("nonsense")).isNull();
  }

  @Test
  public void testIsAttributeExplicitlySpecified() throws Exception {
    assertThat(mapper.isAttributeValueExplicitlySpecified("srcs")).isTrue();
    assertThat(mapper.isAttributeValueExplicitlySpecified("deps")).isFalse();
    assertThat(mapper.isAttributeValueExplicitlySpecified("nonsense")).isFalse();
  }

  @Test
  public void testVisitation() throws Exception {
    assertThat(getLabelsForAttribute(mapper, "srcs")).containsExactly("//p:a", "//p:b", "//p:c");
  }

  protected static List<String> getLabelsForAttribute(
      AttributeMap attributeMap, String attributeName) throws InterruptedException {
    List<String> labels = new ArrayList<>();
    attributeMap.visitLabels(
        attributeMap.getAttributeDefinition(attributeName), label -> labels.add(label.toString()));
    return labels;
  }
}
