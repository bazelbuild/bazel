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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.HashMap;
import java.util.Map;

@RunWith(JUnit4.class)
public class RuleFactoryTest extends PackageLoadingTestCase {

  private ConfiguredRuleClassProvider provider = TestRuleClassProvider.getRuleClassProvider();
  private RuleFactory ruleFactory = new RuleFactory(provider);

  public static final Location LOCATION_42 = Location.fromFileAndOffsets(null, 42, 42);

  @Test
  public void testCreateRule() throws Exception {
    Path myPkgPath = scratch.resolve("/foo/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder =
        new Package.Builder(PackageIdentifier.createInDefaultRepo("mypkg"), "TESTING")
            .setFilename(myPkgPath)
            .setMakeEnv(new MakeEnvironment.Builder());

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("alwayslink", true);

    Rule rule =
        RuleFactory.createAndAddRule(
            pkgBuilder,
            provider.getRuleClassMap().get("cc_library"),
            new BuildLangTypedAttributeValuesMap(attributeValues),
            new Reporter(),
            /*ast=*/ null,
            LOCATION_42,
            /*env=*/ null);

    assertSame(rule, rule.getAssociatedRule());

    // pkg.getRules() = [rule]
    Package pkg = pkgBuilder.build();
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class))).hasSize(1);
    assertEquals(rule, pkg.getTargets(Rule.class).iterator().next());

    assertSame(rule, pkg.getTarget("foo"));

    assertEquals(Label.parseAbsolute("//mypkg:foo"), rule.getLabel());
    assertEquals("foo", rule.getName());

    assertEquals("cc_library", rule.getRuleClass());
    assertEquals("cc_library rule", rule.getTargetKind());
    assertEquals(42, rule.getLocation().getStartOffset());
    assertFalse(rule.containsErrors());

    // Attr with explicitly-supplied value:
    AttributeMap attributes = RawAttributeMapper.of(rule);
    assertTrue(attributes.get("alwayslink", Type.BOOLEAN));
    try {
      attributes.get("alwayslink", Type.STRING); // type error: boolean, not string!
      fail();
    } catch (Exception e) {
      /* Class of exception and error message are not specified by API. */
    }
    try {
      attributes.get("nosuchattr", Type.STRING); // no such attribute
      fail();
    } catch (Exception e) {
      /* Class of exception and error message are not specified by API. */
    }

    // Attrs with default values:
    // cc_library linkstatic default=0 according to build encyc.
    assertFalse(attributes.get("linkstatic", Type.BOOLEAN));
    assertFalse(attributes.get("testonly", Type.BOOLEAN));
    assertThat(attributes.get("srcs", BuildType.LABEL_LIST)).isEmpty();
  }

  @Test
  public void testCreateWorkspaceRule() throws Exception {
    Path myPkgPath = scratch.resolve("/foo/workspace/WORKSPACE");
    Package.Builder pkgBuilder = Package.newExternalPackageBuilder(myPkgPath, "TESTING");

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//foo:bar");

    Rule rule =
        RuleFactory.createAndAddRule(
            pkgBuilder,
            provider.getRuleClassMap().get("bind"),
            new BuildLangTypedAttributeValuesMap(attributeValues),
            new Reporter(),
            /*ast=*/ null,
            Location.fromFileAndOffsets(myPkgPath.asFragment(), 42, 42),
            /*env=*/ null);
    assertFalse(rule.containsErrors());
  }

  @Test
  public void testWorkspaceRuleFailsInBuildFile() throws Exception {
    Path myPkgPath = scratch.resolve("/foo/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder =
        new Package.Builder(PackageIdentifier.createInDefaultRepo("mypkg"), "TESTING")
            .setFilename(myPkgPath)
            .setMakeEnv(new MakeEnvironment.Builder());

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//bar:baz");

    try {
      RuleFactory.createAndAddRule(
          pkgBuilder,
          provider.getRuleClassMap().get("bind"),
          new BuildLangTypedAttributeValuesMap(attributeValues),
          new Reporter(),
          /*ast=*/ null,
          LOCATION_42,
          /*env=*/ null);
      fail();
    } catch (RuleFactory.InvalidRuleException e) {
      assertThat(e.getMessage()).contains("must be in the WORKSPACE file");
    }
  }

  @Test
  public void testBuildRuleFailsInWorkspaceFile() throws Exception {
    Path myPkgPath = scratch.resolve("/foo/workspace/WORKSPACE");
    Package.Builder pkgBuilder =
        new Package.Builder(Label.EXTERNAL_PACKAGE_IDENTIFIER, "TESTING")
            .setFilename(myPkgPath)
            .setMakeEnv(new MakeEnvironment.Builder());

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("alwayslink", true);

    try {
      RuleFactory.createAndAddRule(
          pkgBuilder,
          provider.getRuleClassMap().get("cc_library"),
          new BuildLangTypedAttributeValuesMap(attributeValues),
          new Reporter(),
          /*ast=*/ null,
          Location.fromFileAndOffsets(myPkgPath.asFragment(), 42, 42),
          /*env=*/ null);
      fail();
    } catch (RuleFactory.InvalidRuleException e) {
      assertThat(e.getMessage()).contains("cannot be in the WORKSPACE file");
    }
  }

  private void assertAttr(RuleClass ruleClass, String attrName, Type<?> type) throws Exception {
    assertTrue(
        "Rule class '"
            + ruleClass.getName()
            + "' should have attribute '"
            + attrName
            + "' of type '"
            + type
            + "'",
        ruleClass.hasAttr(attrName, type));
  }

  @Test
  public void testOutputFileNotEqualDot() throws Exception {
    Path myPkgPath = scratch.resolve("/foo");
    Package.Builder pkgBuilder =
        new Package.Builder(PackageIdentifier.createInDefaultRepo("mypkg"), "TESTING")
            .setFilename(myPkgPath)
            .setMakeEnv(new MakeEnvironment.Builder());

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("outs", Lists.newArrayList("."));
    attributeValues.put("name", "some");
    try {
      RuleFactory.createAndAddRule(
          pkgBuilder,
          provider.getRuleClassMap().get("genrule"),
          new BuildLangTypedAttributeValuesMap(attributeValues),
          new Reporter(),
          /*ast=*/ null,
          Location.fromFileAndOffsets(myPkgPath.asFragment(), 42, 42),
          /*env=*/ null);
      fail();
    } catch (RuleFactory.InvalidRuleException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("output file name can't be equal '.'"));
    }
  }

  /**
   * Tests mandatory attribute definitions for test rules.
   */
  // TODO(ulfjack): Remove this check when we switch over to the builder
  // pattern, which will always guarantee that these attributes are present.
  @Test
  public void testTestRules() throws Exception {
    Path myPkgPath = scratch.resolve("/foo/workspace/mypkg/BUILD");
    Package pkg =
        new Package.Builder(PackageIdentifier.createInDefaultRepo("mypkg"), "TESTING")
            .setFilename(myPkgPath)
            .setMakeEnv(new MakeEnvironment.Builder())
            .build();

    for (String name : ruleFactory.getRuleClassNames()) {
      // Create rule instance directly so we'll avoid mandatory attribute check yet will be able
      // to use TargetUtils.isTestRule() method to identify test rules.
      RuleClass ruleClass = ruleFactory.getRuleClass(name);
      Rule rule =
          new Rule(
              pkg,
              pkg.createLabel("myrule"),
              ruleClass,
              Location.fromFile(myPkgPath),
              new AttributeContainer(ruleClass));
      if (TargetUtils.isTestRule(rule)) {
        assertAttr(ruleClass, "tags", Type.STRING_LIST);
        assertAttr(ruleClass, "size", Type.STRING);
        assertAttr(ruleClass, "flaky", Type.BOOLEAN);
        assertAttr(ruleClass, "shard_count", Type.INTEGER);
        assertAttr(ruleClass, "local", Type.BOOLEAN);
      }
    }
  }
}
