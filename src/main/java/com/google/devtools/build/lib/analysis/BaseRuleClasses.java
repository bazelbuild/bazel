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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.constraints.ConstraintConstants;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LabelListLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault.Resolver;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule class definitions used by (almost) every rule.
 */
public class BaseRuleClasses {

  private BaseRuleClasses() {}

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Attribute.ComputedDefault testonlyDefault =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          return rule.getPackageDefaultTestOnly();
        }
      };

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Attribute.ComputedDefault deprecationDefault =
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
  @AutoCodec @VisibleForTesting
  static final LabelListLateBoundDefault<?> ACTION_LISTENER =
      LabelListLateBoundDefault.fromTargetConfiguration(
          BuildConfiguration.class,
          (rule, attributes, configuration) -> configuration.getActionListeners());

  public static final String DEFAULT_COVERAGE_SUPPORT_VALUE = "//tools/test:coverage_support";

  @AutoCodec
  static final Resolver<TestConfiguration, Label> COVERAGE_SUPPORT_CONFIGURATION_RESOLVER =
      (rule, attributes, configuration) -> configuration.getCoverageSupport();

  public static LabelLateBoundDefault<TestConfiguration> coverageSupportAttribute(
      Label defaultValue) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        TestConfiguration.class, defaultValue, COVERAGE_SUPPORT_CONFIGURATION_RESOLVER);
  }

  public static final String DEFAULT_COVERAGE_REPORT_GENERATOR_VALUE =
      "//tools/test:coverage_report_generator";

  private static final String DEFAULT_COVERAGE_OUTPUT_GENERATOR_VALUE =
      "@bazel_tools//tools/test:lcov_merger";

  @AutoCodec
  static final Resolver<TestConfiguration, Label> COVERAGE_REPORT_GENERATOR_CONFIGURATION_RESOLVER =
      (rule, attributes, configuration) -> configuration.getCoverageReportGenerator();

  public static LabelLateBoundDefault<TestConfiguration> coverageReportGeneratorAttribute(
      Label defaultValue) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        TestConfiguration.class, defaultValue, COVERAGE_REPORT_GENERATOR_CONFIGURATION_RESOLVER);
  }

  public static LabelLateBoundDefault<BuildConfiguration> getCoverageOutputGeneratorLabel() {
    return LabelLateBoundDefault.fromTargetConfiguration(
        BuildConfiguration.class, null, COVERAGE_OUTPUT_GENERATOR_RESOLVER);
  }

  @AutoCodec
  static final Resolver<BuildConfiguration, Label> COVERAGE_OUTPUT_GENERATOR_RESOLVER =
      (rule, attributes, configuration) -> {
        if (configuration.isCodeCoverageEnabled()) {
          return Label.parseAbsoluteUnchecked(DEFAULT_COVERAGE_OUTPUT_GENERATOR_VALUE);
        } else {
          return null;
        }
      };

  // TODO(b/65746853): provide a way to do this without passing the entire configuration
  /** Implementation for the :run_under attribute. */
  @AutoCodec
  public static final LabelLateBoundDefault<?> RUN_UNDER =
      LabelLateBoundDefault.fromTargetConfiguration(
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
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(TestConfiguration.class)
          // TestConfiguration only needed to create TestAction and TestProvider
          // Only necessary at top-level and can be skipped if trimmed.
          .setMissingFragmentPolicy(TestConfiguration.class, MissingFragmentPolicy.IGNORE)
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
              attr("$test_wrapper", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(env.getToolsLabel("//tools/test:test_wrapper")))
          .add(
              attr("$xml_writer", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(env.getToolsLabel("//tools/test:xml_writer")))
          .add(
              attr("$test_runtime", LABEL_LIST)
                  .cfg(HostTransition.createFactory())
                  .value(getTestRuntimeLabelList(env)))
          .add(
              attr("$test_setup_script", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(env.getToolsLabel("//tools/test:test_setup")))
          .add(
              attr("$xml_generator_script", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(env.getToolsLabel("//tools/test:test_xml_generator")))
          .add(
              attr("$collect_coverage_script", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(env.getToolsLabel("//tools/test:collect_coverage")))
          // Input files for test actions collecting code coverage
          .add(
              attr(":coverage_support", LABEL)
                  .value(
                      coverageSupportAttribute(env.getToolsLabel(DEFAULT_COVERAGE_SUPPORT_VALUE))))
          // Used in the one-per-build coverage report generation action.
          .add(
              attr(":coverage_report_generator", LABEL)
                  .cfg(HostTransition.createFactory())
                  .value(
                      coverageReportGeneratorAttribute(
                          env.getToolsLabel(DEFAULT_COVERAGE_REPORT_GENERATOR_VALUE))))
          // The target itself and run_under both run on the same machine.
          .add(attr(":run_under", LABEL).value(RUN_UNDER).skipPrereqValidatorCheck())
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

  private static final String TOOLS_TEST_RUNTIME_TARGET_PATTERN = "//tools/test:runtime";
  private static ImmutableList<Label> testRuntimeLabelList = null;

  // Always return the same ImmutableList<Label> for every $test_runtime attribute's default value.
  public static synchronized ImmutableList<Label> getTestRuntimeLabelList(
      RuleDefinitionContext env) {
    if (testRuntimeLabelList == null) {
      testRuntimeLabelList =
          ImmutableList.of(
              Label.parseAbsoluteUnchecked(
                  env.getToolsRepository() + TOOLS_TEST_RUNTIME_TARGET_PATTERN));
    }
    return testRuntimeLabelList;
  }

  /**
   * The attribute used to list the configuration properties used by a target and its transitive
   * dependencies. Currently only supports config_feature_flag.
   */
  public static final String TAGGED_TRIMMING_ATTR = "transitive_configs";

  /** Share common attributes across both base and Starlark base rules. */
  public static RuleClass.Builder commonCoreAndStarlarkAttributes(RuleClass.Builder builder) {
    return builder
        // The visibility attribute is special: it is a nodep label, and loading the
        // necessary package groups is handled by {@link LabelVisitor#visitTargetVisibility}.
        // Package groups always have the null configuration so that they are not duplicated
        // needlessly.
        .add(
            attr("visibility", NODEP_LABEL_LIST)
                .orderIndependent()
                .cfg(HostTransition.createFactory())
                .nonconfigurable(
                    "special attribute integrated more deeply into Bazel's core logic"))
        .add(
            attr(TAGGED_TRIMMING_ATTR, NODEP_LABEL_LIST)
                .orderIndependent()
                .nonconfigurable("Used in determining configuration"))
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
        .add(
            attr(":action_listener", LABEL_LIST)
                .cfg(HostTransition.createFactory())
                .value(ACTION_LISTENER))
        .add(
            attr(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
                .allowedRuleClasses(ConstraintConstants.ENVIRONMENT_RULE)
                .cfg(HostTransition.createFactory())
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .dontCheckConstraints()
                .nonconfigurable(
                    "special logic for constraints and select: see ConstraintSemantics"))
        .add(
            attr(RuleClass.RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
                .allowedRuleClasses(ConstraintConstants.ENVIRONMENT_RULE)
                .cfg(HostTransition.createFactory())
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .dontCheckConstraints()
                .nonconfigurable(
                    "special logic for constraints and select: see ConstraintSemantics"))
        .add(
            attr(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE, LABEL_LIST)
                .nonconfigurable("stores configurability keys"))
        .add(
            attr(RuleClass.APPLICABLE_LICENSES_ATTR, LABEL_LIST)
                .cfg(HostTransition.createFactory())
                .allowedFileTypes(FileTypeSet.NO_FILE)
                // TODO(b/148601291): Require provider to be "LicenseInfo".
                .dontCheckConstraints()
                .nonconfigurable("applicable_licenses is not configurable"));
  }

  public static RuleClass.Builder nameAttribute(RuleClass.Builder builder) {
    return builder.add(attr("name", STRING).nonconfigurable("Rule name"));
  }

  public static RuleClass.Builder execPropertiesAttribute(RuleClass.Builder builder)
      throws ConversionException {
    return builder.add(
        attr(RuleClass.EXEC_PROPERTIES, STRING_DICT).defaultValue(ImmutableMap.of()));
  }

  /**
   * Ancestor of every rule.
   *
   * <p>Adds the name attribute to every rule.
   */
  public static final class RootRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
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
      return commonCoreAndStarlarkAttributes(builder)
          .add(
              attr("licenses", LICENSE)
                  .nonconfigurable("Used in core loading phase logic with no access to configs"))
          .add(
              attr("distribs", DISTRIBUTIONS)
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
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          // Documented in
          // com/google/devtools/build/docgen/templates/attributes/common/toolchains.html.
          .add(
              attr("toolchains", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .mandatoryProviders(ImmutableList.of(TemplateVariableInfo.PROVIDER.id()))
                  .dontCheckConstraints())
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
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("deps", LABEL_LIST).legacyAllowAnyFileType())
          .add(
              attr("data", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .dontCheckConstraints())
          .add(attr(RuleClass.EXEC_PROPERTIES, Type.STRING_DICT).value(ImmutableMap.of()))
          .add(
              attr(RuleClass.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)
                  .allowedFileTypes()
                  .nonconfigurable("Used in toolchain resolution")
                  .value(ImmutableList.of()))
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
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
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
}
