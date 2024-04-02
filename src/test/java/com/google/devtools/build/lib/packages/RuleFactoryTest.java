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
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class RuleFactoryTest extends PackageLoadingTestCase {

  private final ConfiguredRuleClassProvider provider = TestRuleClassProvider.getRuleClassProvider();

  private static final ImmutableList<StarlarkThread.CallStackEntry> DUMMY_STACK =
      ImmutableList.of(
          StarlarkThread.callStackEntry(
              StarlarkThread.TOP_LEVEL, Location.fromFileLineColumn("BUILD", 42, 1)),
          StarlarkThread.callStackEntry("foo", Location.fromFileLineColumn("foo.bzl", 10, 1)),
          StarlarkThread.callStackEntry("myrule", Location.fromFileLineColumn("bar.bzl", 30, 6)));

  private Package.Builder newBuilder(PackageIdentifier id, Path filename) {
    return packageFactory
        .newPackageBuilder(
            id,
            RootedPath.toRootedPath(root, filename),
            "TESTING",
            Optional.empty(),
            Optional.empty(),
            StarlarkSemantics.DEFAULT,
            /* repositoryMapping= */ RepositoryMapping.ALWAYS_FALLBACK,
            /* cpuBoundSemaphore= */ null,
            /* generatorMap= */ null,
            /* configSettingVisibilityPolicy= */ null,
            /* globber= */ null)
        .setLoads(ImmutableList.of());
  }

  @Test
  public void testCreateRule(@TestParameter boolean explicitlySetGeneratorAttrs) throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder = newBuilder(PackageIdentifier.createInMainRepo("mypkg"), myPkgPath);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("executable", true);
    attributeValues.put("outs", ImmutableList.of("foo.out"));
    attributeValues.put("cmd", "echo");

    // TODO(b/274802222): Should this be prohibited?
    if (explicitlySetGeneratorAttrs) {
      attributeValues.put("generator_name", "fake_generator_name");
      attributeValues.put("generator_function", "fake_generator_function");
    }

    RuleClass ruleClass = provider.getRuleClassMap().get("genrule");
    Rule rule =
        RuleFactory.createAndAddRule(
            pkgBuilder,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(attributeValues),
            true,
            DUMMY_STACK);

    assertThat(rule.getAssociatedRule()).isSameInstanceAs(rule);

    // pkg.getRules() = [rule]
    Package pkg = pkgBuilder.build();
    assertThat(Sets.newHashSet(pkg.getTargets(Rule.class))).hasSize(1);
    assertThat(pkg.getTargets(Rule.class).iterator().next()).isEqualTo(rule);

    assertThat(pkg.getTarget("foo")).isSameInstanceAs(rule);

    assertThat(rule.getLabel()).isEqualTo(Label.parseCanonical("//mypkg:foo"));
    assertThat(rule.getName()).isEqualTo("foo");

    assertThat(rule.getRuleClass()).isEqualTo("genrule");
    assertThat(rule.getTargetKind()).isEqualTo("genrule rule");
    // The rule reports the location of the outermost call (aka generator), in the BUILD file.
    // This behavior was added to fix b/23974287, but it loses information and is redundant
    // w.r.t. generator_location. A better fix to that issue would be to keep rule.location as
    // the innermost call, and to report the entire call stack at the first error for the rule.
    assertThat(rule.getLocation().file()).isEqualTo("BUILD");
    assertThat(rule.getLocation().line()).isEqualTo(42);
    assertThat(rule.getLocation().column()).isEqualTo(1);
    assertThat(rule.containsErrors()).isFalse();

    // Attr with explicitly-supplied value:
    AttributeMap attributes = RawAttributeMapper.of(rule);
    assertThat(attributes.get("executable", Type.BOOLEAN)).isTrue();
    assertThrows(Exception.class, () -> attributes.get("tools", Type.STRING));
    assertThrows(Exception.class, () -> attributes.get("nosuchattr", Type.STRING));

    // Attrs with default values:
    // cc_library linkstatic default=0 according to build encyc.
    assertThat(attributes.get("output_to_bindir", Type.BOOLEAN)).isFalse();
    assertThat(attributes.get("testonly", Type.BOOLEAN)).isFalse();
    assertThat(attributes.get("srcs", BuildType.LABEL_LIST)).isEmpty();
  }

  @Test
  public void testCreateWorkspaceRule() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/WORKSPACE");
    Package.Builder pkgBuilder =
        packageFactory.newExternalPackageBuilder(
            WorkspaceFileValue.key(RootedPath.toRootedPath(root, myPkgPath)),
            "TESTING",
            RepositoryMapping.ALWAYS_FALLBACK,
            StarlarkSemantics.DEFAULT);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//foo:bar");

    RuleClass ruleClass = provider.getRuleClassMap().get("bind");
    Rule rule =
        RuleFactory.createAndAddRule(
            pkgBuilder,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(attributeValues),
            true,
            DUMMY_STACK);
    assertThat(rule.containsErrors()).isFalse();
  }

  @Test
  public void testWorkspaceRuleFailsInBuildFile() {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package.Builder pkgBuilder = newBuilder(PackageIdentifier.createInMainRepo("mypkg"), myPkgPath);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("actual", "//bar:baz");

    RuleClass ruleClass = provider.getRuleClassMap().get("bind");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRule(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    true,
                    DUMMY_STACK));
    assertThat(e).hasMessageThat().contains("must be in the WORKSPACE file");
  }

  @Test
  public void testBuildRuleFailsInWorkspaceFile() {
    Path myPkgPath = scratch.resolve("/workspace/WORKSPACE");
    Package.Builder pkgBuilder = newBuilder(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, myPkgPath);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("name", "foo");
    attributeValues.put("alwayslink", true);

    RuleClass ruleClass = provider.getRuleClassMap().get("cc_library");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRule(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    true,
                    DUMMY_STACK));
    assertThat(e).hasMessageThat().contains("cannot be in the WORKSPACE file");
  }

  private static void assertAttr(RuleClass ruleClass, String attrName, Type<?> type) {
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
  public void testOutputFileNotEqualDot() {
    Path myPkgPath = scratch.resolve("/workspace/mypkg");
    Package.Builder pkgBuilder = newBuilder(PackageIdentifier.createInMainRepo("mypkg"), myPkgPath);

    Map<String, Object> attributeValues = new HashMap<>();
    attributeValues.put("outs", Lists.newArrayList("."));
    attributeValues.put("name", "some");
    RuleClass ruleClass = provider.getRuleClassMap().get("genrule");
    RuleFactory.InvalidRuleException e =
        assertThrows(
            RuleFactory.InvalidRuleException.class,
            () ->
                RuleFactory.createAndAddRule(
                    pkgBuilder,
                    ruleClass,
                    new BuildLangTypedAttributeValuesMap(attributeValues),
                    true,
                    DUMMY_STACK));
    assertWithMessage(e.getMessage())
        .that(e.getMessage().contains("output file name can't be equal '.'"))
        .isTrue();
  }

  /** Tests mandatory attribute definitions for test rules. */
  // TODO(ulfjack): Remove this check when we switch over to the builder
  // pattern, which will always guarantee that these attributes are present.
  @Test
  public void testTestRules() throws Exception {
    Path myPkgPath = scratch.resolve("/workspace/mypkg/BUILD");
    Package pkg = newBuilder(PackageIdentifier.createInMainRepo("mypkg"), myPkgPath).build();

    for (RuleClass ruleClass : provider.getRuleClassMap().values()) {
      // Create rule instance directly so we'll avoid mandatory attribute check yet will be able
      // to use TargetUtils.isTestRule() method to identify test rules.
      Rule rule =
          new Rule(
              pkg,
              Label.create(pkg.getPackageIdentifier(), "myrule"),
              ruleClass,
              Location.fromFile(myPkgPath.toString()),
              /* interiorCallStack= */ null);
      if (TargetUtils.isTestRule(rule)) {
        assertAttr(ruleClass, "tags", Types.STRING_LIST);
        assertAttr(ruleClass, "size", Type.STRING);
        assertAttr(ruleClass, "flaky", Type.BOOLEAN);
        assertAttr(ruleClass, "shard_count", Type.INTEGER);
        assertAttr(ruleClass, "local", Type.BOOLEAN);
      }
    }
  }
}
