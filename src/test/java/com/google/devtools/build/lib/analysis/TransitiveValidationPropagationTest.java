// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link OutputGroupInfo#VALIDATION_TRANSITIVE} output group */
@RunWith(JUnit4.class)
public final class TransitiveValidationPropagationTest extends BuildViewTestCase {

  /** Fake native rule that outputs a single validation artifact */
  public static final class ValidationOutputRule
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .addAttribute(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.NO_FILE).build())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("validation_rule")
          .ancestors(BaseRuleClasses.NativeBuildRule.class)
          .factoryClass(ValidationOutputRule.class)
          .build();
    }

    @Override
    @Nullable
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      Artifact valid = ruleContext.createOutputArtifact();
      ruleContext.registerAction(new NullAction(valid));
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER))
          .addProvider(RunfilesProvider.EMPTY)
          .addOutputGroup(OutputGroupInfo.VALIDATION, valid)
          .build();
    }
  }

  /**
   * Fake native rule that disables transitive validation artifact propagation returning only a
   * single validation artifact
   */
  public static final class TransitiveValidationOverrideRule
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder.build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("transitive_validation_rule")
          .ancestors(BaseRuleClasses.NativeBuildRule.class, ValidationOutputRule.class)
          .factoryClass(TransitiveValidationOverrideRule.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      Artifact valid = ruleContext.createOutputArtifact();
      ruleContext.registerAction(new NullAction(valid));
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER))
          .addProvider(RunfilesProvider.EMPTY)
          .addOutputGroup(OutputGroupInfo.VALIDATION_TRANSITIVE, valid)
          .build();
    }
  }

  /** Make the test rule class provider understand our rules in addition to the standard ones. */
  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(new ValidationOutputRule())
            .addRuleDefinition(new TransitiveValidationOverrideRule());
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void testValidationOutputPropagation() throws Exception {
    scratch.file(
        "valid/BUILD",
        """
        validation_rule(name = "foo")

        validation_rule(
            name = "bar",
            deps = [":foo"],
        )

        validation_rule(name = "baz")

        validation_rule(
            name = "top",
            deps = [
                "bar",
                "baz",
            ],
        )

        transitive_validation_rule(
            name = "top_transitive",
            deps = [
                "bar",
                "baz",
            ],
        )
        """);

    List<String> topValid =
        prettyArtifactNames(
            OutputGroupInfo.get(getConfiguredTarget("//valid:top"))
                .getOutputGroup(OutputGroupInfo.VALIDATION));
    List<String> topTransitiveValid =
        prettyArtifactNames(
            OutputGroupInfo.get(getConfiguredTarget("//valid:top_transitive"))
                .getOutputGroup(OutputGroupInfo.VALIDATION));

    assertThat(topValid).containsExactly("valid/foo", "valid/bar", "valid/baz", "valid/top");
    assertThat(topTransitiveValid).containsExactly("valid/top_transitive");
  }

  @Test
  public void testTransitiveValidationOutputGroupNotAllowedForStarlarkRules() throws Exception {
    scratch.file(
        "test/foo_rule.bzl",
        """
        def _impl(ctx):
            return [OutputGroupInfo(_validation_transitive = depset())]

        foo_rule = rule(implementation = _impl)
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:foo_rule.bzl", "foo_rule")

        foo_rule(name = "foo")
        """);

    AssertionError expected =
        assertThrows(AssertionError.class, () -> getConfiguredTarget("//test:foo"));

    assertThat(expected)
        .hasMessageThat()
        .contains("//test:foo_rule.bzl cannot access the _transitive_validation private API");
  }
}
