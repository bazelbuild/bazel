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
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;

/** Unit tests for classes that extend {@link AbstractAttributeMapper}. */
public abstract class AbstractAttributeMapperTest extends BuildViewTestCase {

  protected Rule rule;
  protected AbstractAttributeMapper mapper;

  protected abstract AbstractAttributeMapper createMapper(Rule rule);

  @Before
  public final void initializeRuleAndMapper() throws Exception {
    rule =
        scratchRule(
            "p",
            "myrule",
            """
            cc_binary(
                name = "myrule",
                srcs = ["a", "b", "c"],
            )
            """);
    mapper = createMapper(rule);
  }

  @Test
  public void testRuleProperties() {
    assertThat(mapper.getLabel().getName()).isEqualTo(rule.getName());
    assertThat(mapper.getLabel()).isEqualTo(rule.getLabel());
  }

  @Test
  public void testPackageDefaultProperties() throws Exception {
    // TODO: blaze-configurability-team - write some package args and test them.
    assertThat(mapper.getPackageArgs()).isEqualTo(rule.getPackage().getPackageArgs());
  }

  @Test
  public void testAttributeTypeChecking() {
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
  public void testIsAttributeExplicitlySpecified() {
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
    attributeMap.visitLabels(attributeName, label -> labels.add(label.toString()));
    return labels;
  }
}
