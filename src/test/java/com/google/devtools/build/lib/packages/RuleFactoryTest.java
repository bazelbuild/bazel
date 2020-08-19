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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RuleFactoryTest extends PackageLoadingTestCase {

  private ConfiguredRuleClassProvider provider = TestRuleClassProvider.getRuleClassProvider();
  private final RuleFactory ruleFactory = new RuleFactory(provider);

  private static final ImmutableList<StarlarkThread.CallStackEntry> DUMMY_STACK =
      ImmutableList.of(
          new StarlarkThread.CallStackEntry(
              "<toplevel>", Location.fromFileLineColumn("BUILD", 42, 1)),
          new StarlarkThread.CallStackEntry("foo", Location.fromFileLineColumn("foo.bzl", 10, 1)),
          new StarlarkThread.CallStackEntry(
              "myrule", Location.fromFileLineColumn("bar.bzl", 30, 6)));

  @Test
  public void testCreateRule() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder =
        packageFactory
            .newPackageBuilder(
                PackageIdentifier.createInMainRepo("mypkg"), "TESTING", StarlarkSemantics.DEFAULT)
            .setFilename(RootedPath.toRootedPath(root, myPkgPath));

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("alwayslink", true);

    RuleClass ruleClass = provider.getRuleClassMap().get("cc_library");
    Rule rule =
        RuleFactory.createAndAddRuleImpl(
            pkgBuilder,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(attributeValues),
            new Reporter(new EventBus()),
            StarlarkSemantics.DEFAULT,
            DUMMY_STACK);

    assertThat(rule.getAssociatedRule()).isSameInstanceAs(rule);

    // pkg.getRules() = [rule]
    Package pkg = pkgBuilder.build();
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class))).hasSize(1);
    assertThat(pkg.getTargets(Rule.class).iterator().next()).isEqualTo(rule);

    assertThat(pkg.getTarget("foo")).isSameInstanceAs(rule);

    assertThat(rule.getLabel()).isEqualTo(Label.parseAbsolute("//mypkg:foo", ImmutableMap.of()));
    assertThat(rule.getName()).isEqualTo("foo");

    assertThat(rule.getRuleClass()).isEqualTo("cc_library");
    assertThat(rule.getTargetKind()).isEqualTo("cc_library rule");
    // The rule reports the location of the outermost call (aka generator), in the BUILD file.
    // Thie behavior was added to fix b/23974287, but it loses informtion and is redundant
    // w.r.t. generator_location. A better fix to that issue would be to keep rule.location as
    // the innermost call, and to report the entire call stack at the first error for the rule.
    assertThat(rule.getLocation().file()).isEqualTo("BUILD");
    assertThat(rule.getLocation().line()).isEqualTo(42);
    assertThat(rule.getLocation().column()).isEqualTo(1);
    assertThat(rule.containsErrors()).isFalse();

    // Attr with explicitly-supplied value:
    AttributeMap attributes = RawAttributeMapper.of(rule);
    assertThat(attributes.get("alwayslink", Type.BOOLEAN)).isTrue();
    assertThrows(Exception.class, () -> attributes.get("alwayslink", Type.STRING));
    assertThrows(Exception.class, () -> attributes.get("nosuchattr", Type.STRING));

    // Attrs with default values:
    // cc_library linkstatic default=0 according to build encyc.
    assertThat(attributes.get("linkstatic", Type.BOOLEAN)).isFalse();
    assertThat(attributes.get("testonly", Type.BOOLEAN)).isFalse();
    assertThat(attributes.get("srcs", BuildType.LABEL_LIST)).isEmpty();
  }

  @Test
  public void testCreateWorkspaceRule() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/WORKSPACE");
    Package.Builder pkgBuilder =
        packageFactory.newExternalPackageBuilder(
            RootedPath.toRootedPath(root, myPkgPath), "TESTING", StarlarkSemantics.DEFAULT);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//foo:bar");

    RuleClass ruleClass = provider.getRuleClassMap().get("bind");
    Rule rule =
        RuleFactory.createAndAddRuleImpl(
            pkgBuilder,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(attributeValues),
            new Reporter(new EventBus()),
            StarlarkSemantics.DEFAULT,
            DUMMY_STACK);
    assertThat(rule.containsErrors()).isFalse();
  }

  @Test
  public void testWorkspaceRuleFailsInBuildFile() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder =
        packageFactory
            .newPackageBuilder(
                PackageIdentifier.createInMainRepo("mypkg"), "TESTING", StarlarkSemantics.DEFAULT)
            .setFilename(RootedPath.toRootedPath(root, myPkgPath));

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//bar:baz");

    RuleClass ruleClass = provider.getRuleClassMap().get("bind");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRuleImpl(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    new Reporter(new EventBus()),
                    StarlarkSemantics.DEFAULT,
                    DUMMY_STACK));
    assertThat(e).hasMessageThat().contains("must be in the WORKSPACE file");
  }

  @Test
  public void testBuildRuleFailsInWorkspaceFile() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/WORKSPACE");
    Package.Builder pkgBuilder =
        packageFactory
            .newPackageBuilder(
                LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, "TESTING", StarlarkSemantics.DEFAULT)
            .setFilename(RootedPath.toRootedPath(root, myPkgPath));

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("alwayslink", true);

    RuleClass ruleClass = provider.getRuleClassMap().get("cc_library");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRuleImpl(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    new Reporter(new EventBus()),
                    StarlarkSemantics.DEFAULT,
                    DUMMY_STACK));
    assertThat(e).hasMessageThat().contains("cannot be in the WORKSPACE file");
  }

  private void assertAttr(RuleClass ruleClass, String attrName, Type<?> type) throws Exception {
    assertWithMessage(
            "Rule class '"
                + ruleClass.getName()
                + "' should have attribute '"
                + attrName
                + "' of type '"
                + type
                + "'")
        .that(ruleClass.hasAttr(attrName, type))
        .isTrue();
  }

  @Test
  public void testOutputFileNotEqualDot() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg");
    Package.Builder pkgBuilder =
        packageFactory
            .newPackageBuilder(
                PackageIdentifier.createInMainRepo("mypkg"), "TESTING", StarlarkSemantics.DEFAULT)
            .setFilename(RootedPath.toRootedPath(root, myPkgPath));

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("outs", Lists.newArrayList("."));
    attributeValues.put("name", "some");
    RuleClass ruleClass = provider.getRuleClassMap().get("genrule");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRuleImpl(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    new Reporter(new EventBus()),
                    StarlarkSemantics.DEFAULT,
                    DUMMY_STACK));
    assertWithMessage(e.getMessage())
        .that(e.getMessage().contains("output file name can't be equal '.'"))
        .isTrue();
  }

  /**
   * Tests mandatory attribute definitions for test rules.
   */
  // TODO(ulfjack): Remove this check when we switch over to the builder
  // pattern, which will always guarantee that these attributes are present.
  @Test
  public void testTestRules() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package pkg =
        packageFactory
            .newPackageBuilder(
                PackageIdentifier.createInMainRepo("mypkg"), "TESTING", StarlarkSemantics.DEFAULT)
            .setFilename(RootedPath.toRootedPath(root, myPkgPath))
            .build();

    for (String name : ruleFactory.getRuleClassNames()) {
      // Create rule instance directly so we'll avoid mandatory attribute check yet will be able
      // to use TargetUtils.isTestRule() method to identify test rules.
      RuleClass ruleClass = ruleFactory.getRuleClass(name);
      Rule rule =
          new Rule(
              pkg,
              Label.create(pkg.getPackageIdentifier(), "myrule"),
              ruleClass,
              Location.fromFile(myPkgPath.toString()),
              CallStack.EMPTY,
              AttributeContainer.newInstance(ruleClass));
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
