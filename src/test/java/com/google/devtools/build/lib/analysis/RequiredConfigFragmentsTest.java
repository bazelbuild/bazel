// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RequiredConfigFragmentsProvider}. */
@RunWith(JUnit4.class)
public final class RequiredConfigFragmentsTest extends BuildViewTestCase {
  @Test
  public void provideTransitiveRequiredFragmentsMode() throws Exception {
    useConfiguration("--include_config_fragments_provider=transitive");
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'config', values = {'start_end_lib': '1'})",
        "py_library(name = 'pylib', srcs = ['pylib.py'])",
        "cc_library(name = 'a', srcs = ['A.cc'], data = [':pylib'])");

    ImmutableSet<String> ccLibTransitiveFragments =
        getConfiguredTarget("//a:a")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(ccLibTransitiveFragments).containsAtLeast("CppConfiguration", "PythonConfiguration");

    ImmutableSet<String> configSettingTransitiveFragments =
        getConfiguredTarget("//a:config")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(configSettingTransitiveFragments).contains("CppOptions");
  }

  @Test
  public void provideDirectRequiredFragmentsMode() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct");
    scratch.file(
        "a/BUILD",
        "config_setting(name = 'config', values = {'start_end_lib': '1'})",
        "py_library(name = 'pylib', srcs = ['pylib.py'])",
        "cc_library(name = 'a', srcs = ['A.cc'], data = [':pylib'])");

    ImmutableSet<String> ccLibDirectFragments =
        getConfiguredTarget("//a:a")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(ccLibDirectFragments).contains("CppConfiguration");
    assertThat(ccLibDirectFragments).doesNotContain("PythonConfiguration");

    ImmutableSet<String> configSettingDirectFragments =
        getConfiguredTarget("//a:config")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(configSettingDirectFragments).contains("CppOptions");
  }

  /**
   * Helper method that returns a combined set of the common fragments all genrules require plus
   * instance-specific requirements passed here.
   */
  private ImmutableSortedSet<String> genRuleFragments(String... targetSpecificRequirements)
      throws Exception {
    scratch.file(
        "base_genrule/BUILD",
        "genrule(",
        "    name = 'base_genrule',",
        "    srcs = [],",
        "    outs = ['base_genrule.out'],",
        "    cmd = 'echo hi > $@')");
    ImmutableSortedSet.Builder<String> builder = ImmutableSortedSet.naturalOrder();
    builder.add(targetSpecificRequirements);
    builder.addAll(
        getConfiguredTarget("//base_genrule")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments());
    return builder.build();
  }

  @Test
  public void requiresMakeVariablesSuppliedByDefine() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct", "--define", "myvar=myval");
    scratch.file(
        "a/BUILD",
        "genrule(",
        "    name = 'myrule',",
        "    srcs = [],",
        "    outs = ['myrule.out'],",
        "    cmd = 'echo $(myvar) $(COMPILATION_MODE) > $@')");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:myrule")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments)
        .containsExactlyElementsIn(genRuleFragments("--define:myvar"))
        .inOrder();
  }

  @Test
  public void starlarkExpandMakeVariables() throws Exception {
    useConfiguration("--include_config_fragments_provider=direct", "--define", "myvar=myval");
    scratch.file(
        "a/defs.bzl",
        "def _impl(ctx):",
        "  print(ctx.expand_make_variables('dummy attribute', 'string with $(myvar)!', {}))",
        "",
        "simple_rule = rule(",
        "  implementation = _impl,",
        "   attrs = {}",
        ")");
    scratch.file("a/BUILD", "load('//a:defs.bzl', 'simple_rule')", "simple_rule(name = 'simple')");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:simple")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments)
        .containsExactlyElementsIn(genRuleFragments("--define:myvar"))
        .inOrder();
  }

  /**
   * Aspect that requires fragments both in its definition and through an optionally set <code>
   * --define custom_define</code>.
   */
  private static final class AspectWithConfigFragmentRequirements extends NativeAspectClass
      implements ConfiguredAspectFactory {
    @Override
    public AspectDefinition getDefinition(AspectParameters params) {
      return new AspectDefinition.Builder(this)
          .requiresConfigurationFragments(JavaConfiguration.class)
          .add(attr("custom_define", Type.STRING).allowedValues(new AllowedValueSet("", "myvar")))
          .build();
    }

    @Override
    public ConfiguredAspect create(
        ConfiguredTargetAndData ctadBase,
        RuleContext ruleContext,
        AspectParameters params,
        String toolsRepository)
        throws ActionConflictException {
      ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(ruleContext);
      String customDefine = ruleContext.attributes().get("custom_define", Type.STRING);
      if (!customDefine.isEmpty()) {
        builder.addRequiredConfigFragments(ImmutableSet.of("--define:" + customDefine));
      }
      return builder.build();
    }
  }

  private static final AspectWithConfigFragmentRequirements
      ASPECT_WITH_CONFIG_FRAGMENT_REQUIREMENTS = new AspectWithConfigFragmentRequirements();

  /** Rule that attaches {@link AspectWithConfigFragmentRequirements} to its deps. */
  public static final class RuleThatAttachesAspect
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("custom_define", Type.STRING).allowedValues(new AllowedValueSet("", "myvar")))
          .add(
              attr("deps", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .aspect(ASPECT_WITH_CONFIG_FRAGMENT_REQUIREMENTS))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_that_attaches_aspect")
          .ancestors(BaseRuleClasses.BaseRule.class)
          .factoryClass(RuleThatAttachesAspect.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext) throws ActionConflictException {
      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.EMPTY)
          .build();
    }
  }

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(new RuleThatAttachesAspect())
            .addNativeAspectClass(ASPECT_WITH_CONFIG_FRAGMENT_REQUIREMENTS);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void aspectDefinitionRequiresFragments() throws Exception {
    scratch.file(
        "a/BUILD",
        "rule_that_attaches_aspect(",
        "    name = 'parent',",
        "    deps = [':dep'])",
        "rule_that_attaches_aspect(",
        "    name = 'dep')");
    useConfiguration("--include_config_fragments_provider=transitive");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:parent")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments).contains("JavaConfiguration");
    assertThat(requiredFragments).doesNotContain("--define:myvar");
  }

  @Test
  public void aspectImplementationRequiresFragments() throws Exception {
    scratch.file(
        "a/BUILD",
        "rule_that_attaches_aspect(",
        "    name = 'parent',",
        "    deps = [':dep'])",
        "rule_that_attaches_aspect(",
        "    name = 'dep',",
        "    custom_define = 'myvar')");
    useConfiguration("--include_config_fragments_provider=transitive");
    ImmutableSet<String> requiredFragments =
        getConfiguredTarget("//a:parent")
            .getProvider(RequiredConfigFragmentsProvider.class)
            .getRequiredConfigFragments();
    assertThat(requiredFragments).contains("JavaConfiguration");
    assertThat(requiredFragments).contains("--define:myvar");
  }
}
