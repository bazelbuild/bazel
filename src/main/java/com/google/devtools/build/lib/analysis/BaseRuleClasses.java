// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransitionProxy.DATA;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.DynamicTransitionMapper;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransitionProxy;
import com.google.devtools.build.lib.analysis.config.transitions.Transition;
import com.google.devtools.build.lib.analysis.constraints.EnvironmentRule;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;

/**
 * Rule class definitions used by (almost) every rule.
 */
public class BaseRuleClasses {
  private static final Attribute.ComputedDefault testonlyDefault =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultTestOnly();
        }
      };

  private static final Attribute.ComputedDefault deprecationDefault =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultDeprecation();
        }
      };

  // TODO(b/65746853): provide a way to do this without passing the entire configuration
  /**
   * Implementation for the :action_listener attribute.
   *
   * <p>action_listeners are special rules; they tell the build system to add extra_actions to
   * existing rules. As such they need an edge to every ConfiguredTarget with the limitation that
   * they only run on the target configuration and should not operate on action_listeners and
   * extra_actions themselves (to avoid cycles).
   */
  @VisibleForTesting
  static final LateBoundDefault<?, List<Label>> ACTION_LISTENER =
      LateBoundDefault.fromTargetConfiguration(
          BuildConfiguration.class,
          ImmutableList.of(),
          (rule, attributes, configuration) -> configuration.getActionListeners());

  // TODO(b/65746853): provide a way to do this without passing the entire configuration
  /** Implementation for the :run_under attribute. */
  public static final LateBoundDefault<?, Label> RUN_UNDER =
      LateBoundDefault.fromTargetConfiguration(
          BuildConfiguration.class,
          null,
          (rule, attributes, configuration) -> {
            RunUnder runUnder = configuration.getRunUnder();
            return runUnder != null ? runUnder.getLabel() : null;
          });

  /**
   * A base rule for all test rules.
   */
  public static final class TestBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(TestConfiguration.class)
          .add(
              attr("size", STRING)
                  .value("medium")
                  .taggable()
                  .nonconfigurable("policy decision: should be consistent across configurations"))
          .add(
              attr("timeout", STRING)
                  .taggable()
                  .nonconfigurable("policy decision: should be consistent across configurations")
                  .value(
                      new Attribute.ComputedDefault() {
                        @Override
                        public Object getDefault(AttributeMap rule) {
                          TestSize size = TestSize.getTestSize(rule.get("size", Type.STRING));
                          if (size != null) {
                            String timeout = size.getDefaultTimeout().toString();
                            if (timeout != null) {
                              return timeout;
                            }
                          }
                          return "illegal";
                        }
                      }))
          .add(
              attr("flaky", BOOLEAN)
                  .value(false)
                  .taggable()
                  .nonconfigurable("policy decision: should be consistent across configurations"))
          .add(attr("shard_count", INTEGER).value(-1))
          .add(
              attr("local", BOOLEAN)
                  .value(false)
                  .taggable()
                  .nonconfigurable("policy decision: should be consistent across configurations"))
          .add(attr("args", STRING_LIST))
          // Input files for every test action
          .add(
              attr("$test_runtime", LABEL_LIST)
                  .cfg(HostTransition.INSTANCE)
                  .value(ImmutableList.of(env.getToolsLabel("//tools/test:runtime"))))
          // Input files for test actions collecting code coverage
          .add(
              attr("$coverage_support", LABEL)
                  .value(env.getLabel("//tools/defaults:coverage_support")))
          // Used in the one-per-build coverage report generation action.
          .add(
              attr("$coverage_report_generator", LABEL)
                  .cfg(HostTransition.INSTANCE)
                  .value(env.getLabel("//tools/defaults:coverage_report_generator"))
                  .singleArtifact())

          // The target itself and run_under both run on the same machine. We use the DATA config
          // here because the run_under acts like a data dependency (e.g. no LIPO optimization).
          .add(attr(":run_under", LABEL).cfg(DATA).value(RUN_UNDER).skipPrereqValidatorCheck())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$test_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(RootRule.class, MakeVariableExpandingRule.class)
          .build();
    }
  }

  /**
   * Share common attributes across both base and Skylark base rules.
   */
  public static RuleClass.Builder commonCoreAndSkylarkAttributes(RuleClass.Builder builder) {
    return PlatformSemantics.platformAttributes(builder)
        // The visibility attribute is special: it is a nodep label, and loading the
        // necessary package groups is handled by {@link LabelVisitor#visitTargetVisibility}.
        // Package groups always have the null configuration so that they are not duplicated
        // needlessly.
        .add(
            attr("visibility", NODEP_LABEL_LIST)
                .orderIndependent()
                .cfg(HostTransition.INSTANCE)
                .nonconfigurable(
                    "special attribute integrated more deeply into Bazel's core logic"))
        .add(
            attr("deprecation", STRING)
                .value(deprecationDefault)
                .nonconfigurable("Used in core loading phase logic with no access to configs"))
        .add(
            attr("tags", STRING_LIST)
                .orderIndependent()
                .taggable()
                .nonconfigurable("low-level attribute, used in TargetUtils without configurations"))
        .add(
            attr("generator_name", STRING)
                .undocumented("internal")
                .nonconfigurable("static structure of a rule"))
        .add(
            attr("generator_function", STRING)
                .undocumented("internal")
                .nonconfigurable("static structure of a rule"))
        .add(
            attr("generator_location", STRING)
                .undocumented("internal")
                .nonconfigurable("static structure of a rule"))
        .add(
            attr("testonly", BOOLEAN)
                .value(testonlyDefault)
                .nonconfigurable("policy decision: rules testability should be consistent"))
        .add(attr("features", STRING_LIST).orderIndependent())
        .add(attr(":action_listener", LABEL_LIST)
            .cfg(HostTransition.INSTANCE)
            .value(ACTION_LISTENER))
        .add(
            attr(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
                .allowedRuleClasses(EnvironmentRule.RULE_NAME)
                .cfg(HostTransition.INSTANCE)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .dontCheckConstraints()
                .nonconfigurable(
                    "special logic for constraints and select: see ConstraintSemantics"))
        .add(
            attr(RuleClass.RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
                .allowedRuleClasses(EnvironmentRule.RULE_NAME)
                .cfg(HostTransition.INSTANCE)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .dontCheckConstraints()
                .nonconfigurable(
                    "special logic for constraints and select: see ConstraintSemantics"));
  }

  public static RuleClass.Builder nameAttribute(RuleClass.Builder builder) {
    return builder.add(attr("name", STRING).nonconfigurable("Rule name"));
  }

  /**
   * Ancestor of every rule.
   *
   * <p>Adds the name attribute to every rule.
   */
  public static final class RootRule implements RuleDefinition {

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
        return nameAttribute(builder).build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$root_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /**
   * Common parts of some rules.
   */
  public static final class BaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return commonCoreAndSkylarkAttributes(builder)
          // Aggregates the labels of all {@link ConfigRuleClasses} rules this rule uses (e.g.
          // keys for configurable attributes). This is specially populated in
          // {@RuleClass#populateRuleAttributeValues}.
          //
          // This attribute is not needed for actual builds. Its main purpose is so query's
          // proto/XML output includes the labels of config dependencies, so, e.g., depserver
          // reverse dependency lookups remain accurate. These can't just be added to the
          // attribute definitions proto/XML queries already output because not all attributes
          // contain labels.
          //
          // Builds and Blaze-interactive queries don't need this because they find dependencies
          // through direct Rule label visitation, which already factors these in.
          .add(attr("$config_dependencies", LABEL_LIST)
              .nonconfigurable("not intended for actual builds"))
          .add(attr("licenses", LICENSE)
              .nonconfigurable("Used in core loading phase logic with no access to configs"))
          .add(attr("distribs", DISTRIBUTIONS)
              .nonconfigurable("Used in core loading phase logic with no access to configs"))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(RootRule.class)
          .build();
    }
  }

  /**
   * A rule that contains a {@code variables=} attribute to allow referencing Make variables.
   */
  public static final class MakeVariableExpandingRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($make_variable_expanding_rule).ATTRIBUTE(toolchains) -->
          The set of toolchains that supply <a href="${link make-variables}">"Make variables"</a>
          that this target can use in some of its attributes. Some rules have toolchains whose Make
          variables they can use by default.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("toolchains", LABEL_LIST)
              .allowedFileTypes(FileTypeSet.NO_FILE)
              .mandatoryProviders(ImmutableList.of(TemplateVariableInfo.PROVIDER.id())))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$make_variable_expanding_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /**
   * Common ancestor class for some rules.
   */
  public static final class RuleBase implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("deps", LABEL_LIST).legacyAllowAnyFileType())
          .add(attr("data", LABEL_LIST).cfg(DATA)
              .allowedFileTypes(FileTypeSet.ANY_FILE)
              .dontCheckConstraints())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  public static final ImmutableSet<String> ALLOWED_RULE_CLASSES =
      ImmutableSet.of("filegroup", "genrule", "Fileset");

  /** A base rule for all binary rules. */
  public static final class BinaryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("args", STRING_LIST))
          .add(attr("output_licenses", LICENSE))
          .add(
              attr("$is_executable", BOOLEAN)
                  .value(true)
                  .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$binary_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(RootRule.class, MakeVariableExpandingRule.class)
          .build();
    }
  }

  /** Rule class for rules in error. */
  public static final class ErrorRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder.publicByDefault().build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$error_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.BaseRule.class)
          .build();
    }
  }

  /**
   * Declares the implementations for {@link ConfigurationTransitionProxy} enums.
   *
   * <p>We can't put this in {@link Attribute} because that's in the {@code lib.packages} package,
   * which has no access to configuration classes.
   *
   * <p>New transitions should extend {@link PatchTransition}, which avoids the need for this map.
   */
  public static final ImmutableMap<Transition, Transition> DYNAMIC_TRANSITIONS_MAP =
      ImmutableMap.of(
          ConfigurationTransitionProxy.NONE, DynamicTransitionMapper.SELF,
          ConfigurationTransitionProxy.NULL, DynamicTransitionMapper.SELF
          // ConfigurationTransitionProxy.DATA is skipped because it's C++-specific.
          // The C++ rule definitions handle its mapping.
      );
}
