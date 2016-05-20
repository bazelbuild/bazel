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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Unit tests for {@link AbstractAttributeMapper}.
 */
@RunWith(JUnit4.class)
public class AbstractAttributeMapperTest extends BuildViewTestCase {

  protected Rule rule;
  protected AbstractAttributeMapper mapper;

  private static class TestMapper extends AbstractAttributeMapper {
    public TestMapper(Package pkg, RuleClass ruleClass, Label ruleLabel,
        AttributeContainer attributes) {
      super(pkg, ruleClass, ruleLabel, attributes);
    }
  }

  @Before
  public final void initializeRuleAndMapper() throws Exception {
    rule = scratchRule("p", "myrule",
        "cc_binary(name = 'myrule',",
        "          srcs = ['a', 'b', 'c'])");
    RuleClass ruleClass = rule.getRuleClassObject();
    mapper =
        new TestMapper(rule.getPackage(), ruleClass, rule.getLabel(), rule.getAttributeContainer());
  }

  @Test
  public void testRuleProperties() throws Exception {
    assertEquals(rule.getName(), mapper.getName());
    assertEquals(rule.getLabel(), mapper.getLabel());
  }

  @Test
  public void testPackageDefaultProperties() throws Exception {
    rule = scratchRule("a", "myrule",
        "cc_binary(name = 'myrule',",
        "          srcs = ['a', 'b', 'c'])");
    Package pkg = rule.getPackage();
    assertEquals(pkg.getDefaultHdrsCheck(), mapper.getPackageDefaultHdrsCheck());
    assertEquals(pkg.getDefaultTestOnly(), mapper.getPackageDefaultTestOnly());
    assertEquals(pkg.getDefaultDeprecation(), mapper.getPackageDefaultDeprecation());
  }

  @Test
  public void testAttributeTypeChecking() throws Exception {
    // Good typing:
    mapper.get("srcs", BuildType.LABEL_LIST);

    // Bad typing:
    try {
      mapper.get("srcs", Type.BOOLEAN);
      fail("Expected type mismatch to trigger an exception");
    } catch (IllegalArgumentException e) {
      // Expected.
    }

    // Unknown attribute:
    try {
      mapper.get("nonsense", Type.BOOLEAN);
      fail("Expected non-existent type to trigger an exception");
    } catch (IllegalArgumentException e) {
      // Expected.
    }
  }

  @Test
  public void testGetAttributeType() throws Exception {
    assertEquals(BuildType.LABEL_LIST, mapper.getAttributeType("srcs"));
    assertNull(mapper.getAttributeType("nonsense"));
  }

  @Test
  public void testGetAttributeDefinition() {
    assertEquals("srcs", mapper.getAttributeDefinition("srcs").getName());
    assertNull(mapper.getAttributeDefinition("nonsense"));
  }

  @Test
  public void testIsAttributeExplicitlySpecified() throws Exception {
    assertTrue(mapper.isAttributeValueExplicitlySpecified("srcs"));
    assertFalse(mapper.isAttributeValueExplicitlySpecified("deps"));
    assertFalse(mapper.isAttributeValueExplicitlySpecified("nonsense"));
  }

  protected static class VisitationRecorder implements AttributeMap.AcceptsLabelAttribute {
    public List<String> labelsVisited = Lists.newArrayList();
    private final String attrName;

    public VisitationRecorder(String attrName) {
      this.attrName = attrName;
    }

    @Override
    public void acceptLabelAttribute(Label label, Attribute attribute) {
      if (attribute.getName().equals(attrName)) {
        labelsVisited.add(label.toString());
      }
    }
  }

  @Test
  public void testVisitation() throws Exception {
    VisitationRecorder recorder = new VisitationRecorder("srcs");
    mapper.visitLabels(recorder);
    assertThat(recorder.labelsVisited).containsExactly("//p:a", "//p:b", "//p:c");
  }

  @Test
  public void testComputedDefault() throws Exception {
    // Should return a valid ComputedDefault instance since this is a computed default:
    assertThat(mapper.getComputedDefault("$stl_default", BuildType.LABEL))
        .isInstanceOf(Attribute.ComputedDefault.class);
    // Should return null since this *isn't* a computed default:
    assertNull(mapper.getComputedDefault("srcs", BuildType.LABEL_LIST));
  }
}
