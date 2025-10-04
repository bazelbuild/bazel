// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.Map;
import net.starlark.java.eval.Sequence;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark aspects {@code attr_aspects} function. */
@RunWith(JUnit4.class)
public final class StarlarkAspectAttrAspectsFunctionTest extends AnalysisTestCase {

  private void createTestDefs(String propagationAttrsFunction, String propagationPredicateFunction)
      throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            AspectInfo = provider()

            def _rule_impl(ctx):
              pass

            simple_rule = rule(
              implementation = _rule_impl,
              attrs = {
                'deps_1': attr.label_list(),
                'deps_2': attr.label_list(),
                '_tool': attr.label(default = Label('//pkg1:rule_tool'))},
            )

            tool_rule = rule(
              implementation = _rule_impl,
            )

            %s

            %s

            def _aspect_impl(target, ctx):
              res = ['cmdline_aspect on ' + str(target.label)]
              rule_attr = ctx.rule.attr
              for dep in (getattr(rule_attr, 'deps_1', []) + getattr(rule_attr, 'deps_2', [])):
                if AspectInfo in dep:
                  res.extend(dep[AspectInfo].res)

              if hasattr(rule_attr, '_tool') and AspectInfo in rule_attr._tool:
                res.extend(rule_attr._tool[AspectInfo].res)

              return [AspectInfo(res = res)]

            cmdline_aspect = aspect(
              implementation = _aspect_impl,
              attr_aspects = _propagation_attrs,
              propagation_predicate = _propagation_predicate,
            )
            """,
            propagationAttrsFunction, propagationPredicateFunction));
  }

  private void createTestDefs(String propagationAttrsFunction) throws Exception {
    createTestDefs(
        propagationAttrsFunction,
        """
        def _propagation_predicate(ctx):
          return True
        """);
  }

  private void createTestPackages() throws Exception {
    scratch.file(
        "config_setting/BUILD", "config_setting(name='defines', values={'define': 'foo=1'})");

    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'simple_rule', 'tool_rule')
        simple_rule(name = 't1', deps_1 = [':d1'], deps_2 = [':d2'])
        simple_rule(name = 'd1', deps_1 = [':d11'], deps_2 = [':d12'])
        simple_rule(name = 'd2', deps_2 = [':d22'])
        simple_rule(name = 'd11')
        simple_rule(name = 'd12')
        simple_rule(name = 'd22')
        tool_rule(name = 'rule_tool')

        simple_rule(
            name = 'with_selects',
            deps_1 = select({
                  '//config_setting:defines': [':d1'],
                  '//conditions:default': ['d2'],
                })
        )

        simple_rule(name = 't2', deps_1 = [':d1', ':d11'], deps_2 = [':d2'])
        """);
  }

  @Test
  public void selectedAttrsReturned_aspectPropagatesAlongReturnedAttrs() throws Exception {
    createTestDefs(
        """
        def _propagation_attrs(ctx):
          if ctx.rule.label == Label('//pkg1:t1'):
            attr_aspects = []
            for attr_name in dir(ctx.rule.attr):
              attr_value = getattr(ctx.rule.attr, attr_name).value
              if type(attr_value) == type(Label("//foo")):
                attr_aspects.append(attr_name)
              if type(attr_value) == type([]):
                if len(attr_value) > 0 and type(attr_value[0]) == type(Label("//foo")):
                  attr_aspects.append(attr_name)
            return attr_aspects

          else:
            return []
        """);
    createTestPackages();

    var analysisResult = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:t1");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:t1",
            "cmdline_aspect on //pkg1:d1",
            "cmdline_aspect on //pkg1:d2",
            "cmdline_aspect on //pkg1:rule_tool");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:t1",
            "cmdline_aspect on @@//pkg1:d1",
            "cmdline_aspect on @@//pkg1:d2",
            "cmdline_aspect on @@//pkg1:rule_tool");
  }

  @Test
  public void wildCardAttrReturned_buildFails() throws Exception {
    createTestDefs(
        """
        def _propagation_attrs(ctx):
            return ['*']
        """);
    createTestPackages();

    reporter.removeHandler(failFastHandler);
    assertThrows(
        ViewCreationFailedException.class,
        () -> update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:t1"));
    assertContainsEvent("'*' is not allowed in 'attr_aspects' list");
  }

  @Test
  public void invalidReturnValue_buildFails() throws Exception {
    createTestDefs(
        """
        def _propagation_attrs(ctx):
            return [44, 'foo']
        """);
    createTestPackages();

    reporter.removeHandler(failFastHandler);
    assertThrows(
        ViewCreationFailedException.class,
        () -> update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:t1"));
    assertContainsEvent("at index 0 of attr_aspects, got element of type int, want string");
  }

  @Test
  public void withPropagationPredicate_aspectPropagatedToSatisfyingTargets() throws Exception {
    createTestDefs(
        """
        def _propagation_attrs(ctx):
          return ['deps_1']
        """,
        """
        def _propagation_predicate(ctx):
          if ctx.rule.label == Label('//pkg1:d1'):
            return False
          return True
        """);
    createTestPackages();

    var analysisResult = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:t2");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly("cmdline_aspect on //pkg1:t2", "cmdline_aspect on //pkg1:d11");

    var aspectAResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectAResult)
        .containsExactly("cmdline_aspect on @@//pkg1:t2", "cmdline_aspect on @@//pkg1:d11");
  }

  @Test
  public void aspectApplyToGeneratingRule_attrAspectsFuncRunOnGeneratingRule() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AspectInfo = provider()

        def _propagation_attrs(ctx):
          if ctx.rule.label == Label('//pkg1:target_with_output'):
            if ctx.rule.attr.out.value == Label('//pkg1:my_out.txt'):
              return ['deps']
          return []

        def _aspect_impl(target, ctx):
          res = ['generating_rule_aspect on ' + str(target.label)]
          rule_attr = ctx.rule.attr
          for dep in getattr(rule_attr, 'deps', []):
            if AspectInfo in dep:
              res.extend(dep[AspectInfo].res)
          return [AspectInfo(res = res)]

        generating_rule_aspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = _propagation_attrs,
            apply_to_generating_rules = True,
        )

        def _out_rule_impl(ctx):
          if ctx.outputs.out:
            ctx.actions.write(ctx.outputs.out, 'hi')
          return []

        out_rule = rule(
          implementation = _out_rule_impl,
          attrs = {'deps': attr.label_list(), 'out': attr.output()},
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'out_rule')
        out_rule(name = 'target_with_output', out = 'my_out.txt', deps = [':d1'])
        out_rule(name = 'd1')
        """);

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%generating_rule_aspect"), "//pkg1:my_out.txt");

    assertThat(getFormattedAspectKeys("generating_rule_aspect"))
        .containsExactly(
            "generating_rule_aspect on //pkg1:my_out.txt",
            "generating_rule_aspect on //pkg1:target_with_output",
            "generating_rule_aspect on //pkg1:d1");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "generating_rule_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "generating_rule_aspect on @@//pkg1:target_with_output",
            "generating_rule_aspect on @@//pkg1:d1");
  }

  @Test
  public void attrWithSelect_availableToAttrAspectsFunc() throws Exception {
    createTestDefs(
        """
        def _propagation_attrs(ctx):
          deps_1 = ctx.rule.attr.deps_1.value
          if len(deps_1) == 1 and deps_1[0] == Label('//pkg1:d1'):
            return ['deps_1']
          return []
        """);
    createTestPackages();
    useConfiguration("--define=foo=1");

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:with_selects");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly("cmdline_aspect on //pkg1:with_selects", "cmdline_aspect on //pkg1:d1");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:with_selects", "cmdline_aspect on @@//pkg1:d1");
  }

  @Test
  public void computedDefaultAttr_availableToAttrAspectsFunc() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AspectInfo = provider()

        def _propagation_attrs(ctx):
          if hasattr(ctx.rule.attr, '_computed'):
            cmp = ctx.rule.attr._computed.value

            if Label('//pkg1:my_prefix01') in cmp and Label('//pkg1:my_prefix02') in cmp:
              return ['_computed']

          return []

        def _aspect_impl(target, ctx):
          res = ['cmdline_aspect on ' + str(target.label)]
          rule_attr = ctx.rule.attr
          for dep in getattr(rule_attr, '_computed', []):
            if AspectInfo in dep:
              res.extend(dep[AspectInfo].res)
          return [AspectInfo(res = res)]

        cmdline_aspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = _propagation_attrs,
        )

        def _compute_attr(prefix):
          return [
              Label("//pkg1:" + prefix + "01"),
              Label("//pkg1:" + prefix + "02"),
          ]

        def _rule_impl(ctx):
          pass

        computed_default_rule = rule(
          implementation = _rule_impl,
          attrs = {
              "prefix": attr.string(default = "default_prefix"),
              "_computed": attr.label_list(default = _compute_attr),
          },
        )

        simple_rule = rule(
          implementation = _rule_impl,
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'computed_default_rule', 'simple_rule')
        computed_default_rule(name = 'computed_default_target', prefix = 'my_prefix')
        simple_rule(name = 'my_prefix01')
        simple_rule(name = 'my_prefix02')
        """);

    var analysisResult =
        update(
            ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:computed_default_target");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:computed_default_target",
            "cmdline_aspect on //pkg1:my_prefix01",
            "cmdline_aspect on //pkg1:my_prefix02");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:computed_default_target",
            "cmdline_aspect on @@//pkg1:my_prefix01",
            "cmdline_aspect on @@//pkg1:my_prefix02");
  }

  /** A custom {@link FragmentOptions} for the latebound attribute test. */
  public static class TestOptions extends FragmentOptions {}

  /** The {@link Fragment} that contains the options. */
  @RequiresOptions(options = {TestOptions.class})
  public static final class TestFragment extends Fragment {
    private final BuildOptions buildOptions;

    public TestFragment(BuildOptions buildOptions) {
      this.buildOptions = buildOptions;
    }

    // Getter required to satisfy AutoCodec.
    public BuildOptions getBuildOptions() {
      return buildOptions;
    }

    public Label getDep() {
      return Label.parseCanonicalUnchecked("//pkg1:latebound_dep");
    }
  }

  @Test
  public void lateBoundAttributes_availableToAttrAspectsFunc() throws Exception {
    RuleDefinition lateBoundDepRule =
        (MockRule)
            () ->
                MockRule.define(
                    "rule_with_latebound_attr",
                    (builder, env) ->
                        builder
                            .add(
                                attr(":latebound_attr", LABEL)
                                    .value(
                                        LabelLateBoundDefault.fromTargetConfiguration(
                                            TestFragment.class,
                                            null,
                                            (rule, attributes, testConfig) -> testConfig.getDep())))
                            .requiresConfigurationFragments(TestFragment.class));

    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(lateBoundDepRule);
    builder.addConfigurationFragment(TestFragment.class);
    useRuleClassProvider(builder.build());

    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AspectInfo = provider()

        def _propagation_attrs(ctx):
          if hasattr(ctx.rule.attr, '_latebound_attr'):
            if ctx.rule.attr._latebound_attr.value == Label('//pkg1:latebound_dep'):
              return ['_latebound_attr']

          return []

        def _aspect_impl(target, ctx):
          res = ['cmdline_aspect on ' + str(target.label)]
          rule_attr = ctx.rule.attr
          if hasattr(rule_attr, '_latebound_attr'):
            if AspectInfo in rule_attr._latebound_attr:
              res.extend(rule_attr._latebound_attr[AspectInfo].res)
          return [AspectInfo(res = res)]


        cmdline_aspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = _propagation_attrs
        )

        def _rule_impl(ctx):
          pass

        simple_rule = rule(
          implementation = _rule_impl,
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'simple_rule')
        rule_with_latebound_attr(name = 'latebound_target')
        simple_rule(name = 'latebound_dep')
        """);

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:latebound_target");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:latebound_target", "cmdline_aspect on //pkg1:latebound_dep");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:latebound_target",
            "cmdline_aspect on @@//pkg1:latebound_dep");
  }

  @Test
  public void toolAttr_correctlyIdentified() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AspectInfo = provider()

        def _propagation_attrs(ctx):
          attr_aspects = []
          for attr_name in dir(ctx.rule.attr):
            if getattr(ctx.rule.attr, attr_name).is_tool:
              attr_aspects.append(attr_name)

          return attr_aspects

        def _aspect_impl(target, ctx):
          res = ['cmdline_aspect on ' + str(target.label)]
          rule_attr = ctx.rule.attr

          if hasattr(rule_attr, '_tool_1') and AspectInfo in rule_attr._tool_1:
            res.extend(rule_attr._tool_1[AspectInfo].res)

          if hasattr(rule_attr, '_tool_2') and AspectInfo in rule_attr._tool_2:
            res.extend(rule_attr._tool_2[AspectInfo].res)

          if hasattr(rule_attr, '_non_tool') and AspectInfo in rule_attr._non_tool:
            res.extend(rule_attr._non_tool[AspectInfo].res)

          return [AspectInfo(res = res)]

        cmdline_aspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = _propagation_attrs
        )

        def _rule_impl(ctx):
          pass

        rule_with_tool = rule(
          implementation = _rule_impl,
          attrs = {
              '_tool_1': attr.label(
                  default = Label('//pkg1:tool_1'), flags = ['IS_TOOL_DEPENDENCY']),
              '_tool_2': attr.label(default = Label('//pkg1:tool_2'), cfg = 'exec'),
              '_non_tool': attr.label(default = Label('//pkg1:non_tool')),
          },
        )

        simple_rule = rule(
          implementation = _rule_impl,
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'rule_with_tool', 'simple_rule')
        rule_with_tool(name = 'main_target')
        simple_rule(name = 'tool_1')
        simple_rule(name = 'tool_2')
        simple_rule(name = 'non_tool')
        """);

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:main_target");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:main_target",
            "cmdline_aspect on //pkg1:tool_1",
            "cmdline_aspect on //pkg1:tool_2");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:main_target",
            "cmdline_aspect on @@//pkg1:tool_1",
            "cmdline_aspect on @@//pkg1:tool_2");
  }

  @Test
  public void attrInitializer_availableToAttrAspectsFunc() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AspectInfo = provider()

        def _propagation_attrs(ctx):
          if hasattr(ctx.rule.attr, 'deps'):
            deps = ctx.rule.attr.deps.value
            if Label('//pkg1:added') in deps and Label('//pkg1:initial') in deps:
              return ['deps']

          return []

        def _aspect_impl(target, ctx):
          res = ['cmdline_aspect on ' + str(target.label)]
          rule_attr = ctx.rule.attr
          for dep in getattr(rule_attr, 'deps', []):
            if AspectInfo in dep:
              res.extend(dep[AspectInfo].res)
          return [AspectInfo(res = res)]

        cmdline_aspect = aspect(
            implementation = _aspect_impl,
            attr_aspects = _propagation_attrs
        )

        def _rule_impl(ctx):
          pass

        def _initializer(name, deps = []):
          return {"deps": deps + ["//pkg1:added"]}

        rule_with_initializer = rule(
          implementation = _rule_impl,
          initializer = _initializer,
          attrs = {
              "deps": attr.label_list(),
          },
        )

        simple_rule = rule(
          implementation = _rule_impl,
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'rule_with_initializer', 'simple_rule')
        rule_with_initializer(name = 'main_target', deps = [":initial"])
        simple_rule(name = 'initial')
        simple_rule(name = 'added')
        """);
    useConfiguration("--experimental_rule_extension_api");

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:main_target");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:main_target",
            "cmdline_aspect on //pkg1:added",
            "cmdline_aspect on //pkg1:initial");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:main_target",
            "cmdline_aspect on @@//pkg1:added",
            "cmdline_aspect on @@//pkg1:initial");
  }

  @Test
  public void aspectOnAspect_eachAspectPropagatesSeparately() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AProv = provider()
        BProv = provider()
        CProv = provider()

        def _a_propagation_attrs(ctx):
          return ['dep_1', 'dep_2']

        def b_propagation_attrs(ctx):
          return ['dep_1']

        def _c_propagation_attrs(ctx):
          return ['dep_2']

        def _aspect_a_impl(target, ctx):
          current_res = 'aspect_a on %s' % target.label
          if BProv in target:
            current_res += ' with BProv'
          if CProv in target:
            current_res += ' with CProv'

          res = [current_res]

          if ctx.rule.attr.dep_1 and AProv in ctx.rule.attr.dep_1:
            res.extend(ctx.rule.attr.dep_1[AProv].res)

          if ctx.rule.attr.dep_2 and AProv in ctx.rule.attr.dep_2:
            res.extend(ctx.rule.attr.dep_2[AProv].res)

          return [AProv(res = res)]

        def _aspect_b_impl(target, ctx):
          return [BProv(res = 'aspect_b on %s' % target.label)]

        def _aspect_c_impl(target, ctx):
          return [CProv(res = 'aspect_c on %s' % target.label)]

        aspect_a = aspect(
          implementation = _aspect_a_impl,
          attr_aspects = _a_propagation_attrs,
          required_aspect_providers = [[BProv], [CProv]],
        )

        aspect_b = aspect(
          implementation = _aspect_b_impl,
          attr_aspects = b_propagation_attrs,
          provides = [BProv],
        )

        aspect_c = aspect(
          implementation = _aspect_c_impl,
          attr_aspects = _c_propagation_attrs,
          provides = [CProv],
        )

        def _rule_impl(ctx):
          return []

        my_rule = rule(
          implementation = _rule_impl,
          attrs = {
              "dep_1": attr.label(),
              "dep_2": attr.label(),
          },
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 'main_target', dep_1 = ':dep_1', dep_2 = ':dep_2')
        my_rule(name = 'dep_1')
        my_rule(name = 'dep_2')
        """);

    var analysisResult =
        update(
            ImmutableList.of(
                "//test:defs.bzl%aspect_c", "//test:defs.bzl%aspect_b", "//test:defs.bzl%aspect_a"),
            "//pkg1:main_target");

    assertThat(getFormattedAspectKeys("aspect_a"))
        .containsExactly(
            "aspect_a on //pkg1:main_target with base aspects: aspect_b,aspect_c",
            "aspect_a on //pkg1:dep_1 with base aspects: aspect_b",
            "aspect_a on //pkg1:dep_2 with base aspects: aspect_c");

    assertThat(getFormattedAspectKeys("aspect_b"))
        .containsExactly("aspect_b on //pkg1:main_target", "aspect_b on //pkg1:dep_1");

    assertThat(getFormattedAspectKeys("aspect_c"))
        .containsExactly("aspect_c on //pkg1:main_target", "aspect_c on //pkg1:dep_2");

    var aspectAResult =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg1:main_target", "AProv");
    assertThat(aspectAResult)
        .containsExactly(
            "aspect_a on @@//pkg1:main_target with BProv with CProv",
            "aspect_a on @@//pkg1:dep_1 with BProv",
            "aspect_a on @@//pkg1:dep_2 with CProv");
  }

  @Test
  public void requiredAspect_propagatesWithMainAspect() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AProv = provider()
        BProv = provider()

        def _a_propagation_attrs(ctx):
          return ['dep_1', 'dep_2']

        def _aspect_a_impl(target, ctx):
          current_res = 'aspect_a on %s' % target.label
          if BProv in target:
            current_res += ' with BProv'

          res = [current_res]

          if ctx.rule.attr.dep_1 and AProv in ctx.rule.attr.dep_1:
            res.extend(ctx.rule.attr.dep_1[AProv].res)

          if ctx.rule.attr.dep_2 and AProv in ctx.rule.attr.dep_2:
            res.extend(ctx.rule.attr.dep_2[AProv].res)

          return [AProv(res = res)]

        def _aspect_b_impl(target, ctx):
          return [BProv(res = 'aspect_b on %s' % target.label)]

        aspect_b = aspect(
          implementation = _aspect_b_impl,
          attr_aspects = [],
          provides = [BProv],
        )

        aspect_a = aspect(
          implementation = _aspect_a_impl,
          attr_aspects = _a_propagation_attrs,
          requires = [aspect_b],
        )

        def _rule_impl(ctx):
          return []

        my_rule = rule(
          implementation = _rule_impl,
          attrs = {
              "dep_1": attr.label(),
              "dep_2": attr.label(),
          },
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 'main_target', dep_1 = ':dep_1', dep_2 = ':dep_2')
        my_rule(name = 'dep_1')
        my_rule(name = 'dep_2')
        """);

    var analysisResult = update(ImmutableList.of("//test:defs.bzl%aspect_a"), "//pkg1:main_target");

    assertThat(getFormattedAspectKeys("aspect_a"))
        .containsExactly(
            "aspect_a on //pkg1:main_target with base aspects: aspect_b",
            "aspect_a on //pkg1:dep_1 with base aspects: aspect_b",
            "aspect_a on //pkg1:dep_2 with base aspects: aspect_b");

    assertThat(getFormattedAspectKeys("aspect_b"))
        .containsExactly(
            "aspect_b on //pkg1:main_target",
            "aspect_b on //pkg1:dep_1",
            "aspect_b on //pkg1:dep_2");

    var aspectAResult =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg1:main_target", "AProv");
    assertThat(aspectAResult)
        .containsExactly(
            "aspect_a on @@//pkg1:main_target with BProv",
            "aspect_a on @@//pkg1:dep_1 with BProv",
            "aspect_a on @@//pkg1:dep_2 with BProv");
  }

  private void createDormantDepsTest(String aspectDef) throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        String.format(
"""
ComponentInfo = provider(fields = ["components"])

def _component_impl(ctx):
  current = struct(label=ctx.label, impl = ctx.attr.impl)
  transitive = [d[ComponentInfo].components for d in ctx.attr.deps]
  return [
    ComponentInfo(components = depset(direct = [current], transitive = transitive)),
  ]

component = rule(
  implementation = _component_impl,
  attrs = {
    "deps": attr.label_list(providers = [ComponentInfo]),
    "impl": attr.dormant_label(),
  },
  provides = [ComponentInfo],
  dependency_resolution_rule = True,
)

def _binary_impl(ctx):
  return [DefaultInfo(files=depset(ctx.files._impls))]

def _materializer(ctx):
  all = depset(transitive = [d[ComponentInfo].components for d in ctx.attr.components])
  selected = [c.impl for c in all.to_list() if "yes" in str(c.label)]
  return selected

binary = rule(
  implementation = _binary_impl,
  attrs = {
      "components": attr.label_list(providers = [ComponentInfo], for_dependency_resolution = True),
      "_impls": attr.label_list(materializer = _materializer),
      "regular_deps": attr.label_list(),
  })

def _rule_impl(ctx):
  pass

simple_rule = rule(
    implementation = _rule_impl,
)

%s
""",
            aspectDef));

    scratch.file(
        "pkg1/BUILD",
"""
load("//test:defs.bzl", "component", "binary", "simple_rule")

component(name="a_yes", impl=":a_impl")
component(name="b_no", deps = [":c_yes", ":d_no"], impl=":b_impl")
component(name="c_yes", impl=":c_impl")
component(name="d_no", impl=":d_impl")

binary(name="bin", components=[":a_yes", ":b_no"], regular_deps = [":dep_1"])
[filegroup(name=x + "_impl", srcs=[x]) for x in ["a", "b", "c", "d"]]

simple_rule(name = 'dep_1')
""");
  }

  @Test
  public void aspectOnMaterializingTarget_attrAspectsFuncUsed() throws Exception {
    createDormantDepsTest(
"""
AspectInfo = provider()

def _propagation_attrs(ctx):
  if ctx.rule.qualified_kind.rule_name == 'binary':
      # the value of _impls should be None as it is not materialized yet
      if ctx.rule.attr._impls.value != None:
        fail("'_impls' should be None")

      # the value of regular_deps should be available
      if Label('//pkg1:dep_1') not in ctx.rule.attr.regular_deps.value:
        fail("regular_deps should be available")

      # the value of components should be available
      components = ctx.rule.attr.components.value
      if len(components) != 2 or Label('//pkg1:a_yes') not in components or Label('//pkg1:b_no') not in components:
        fail("components should be available")

      return ['_impls', 'components', 'regular_deps']

  return []

def _aspect_impl(target, ctx):
  res = ['cmdline_aspect on %s' % target.label]

  deps = []
  deps.extend(getattr(ctx.rule.attr, '_impls', []))
  deps.extend(getattr(ctx.rule.attr, 'components', []))
  deps.extend(getattr(ctx.rule.attr, 'regular_deps', []))

  for dep in deps:
    if AspectInfo in dep:
      res += dep[AspectInfo].res

  return [AspectInfo(res = res)]

cmdline_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = _propagation_attrs,
)
""");

    useConfiguration("--experimental_dormant_deps");
    var analysisResult = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:bin");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:bin",
            "cmdline_aspect on //pkg1:dep_1",
            "cmdline_aspect on //pkg1:a_yes",
            "cmdline_aspect on //pkg1:b_no",
            "cmdline_aspect on //pkg1:a_impl",
            "cmdline_aspect on //pkg1:c_impl");

    var aspectAResult =
        getAspectResult(
            analysisResult.getAspectsMap(), "cmdline_aspect", "//pkg1:bin", "AspectInfo");
    assertThat(aspectAResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:bin",
            "cmdline_aspect on @@//pkg1:dep_1",
            "cmdline_aspect on @@//pkg1:a_yes",
            "cmdline_aspect on @@//pkg1:b_no",
            "cmdline_aspect on @@//pkg1:a_impl",
            "cmdline_aspect on @@//pkg1:c_impl");
  }

  @Test
  public void aspectOnDependencyResolutionTargets_attrAspectsFuncUsed() throws Exception {
    createDormantDepsTest(
"""
AspectInfo = provider()

def _propagation_attrs(ctx):
  if ctx.rule.qualified_kind.rule_name == 'component':
      # the value of impl should be available
      impl_name = ctx.rule.label.name.split('_')[0] + '_impl'
      impl_val = ctx.rule.attr.impl.value
      if type(impl_val) != 'DormantDependency' or impl_val.label.name != impl_name:
        fail("'impl' should be available")

      # the value of deps should be available
      if ctx.rule.label == Label('//pkg1:b_no'):
        deps = ctx.rule.attr.deps.value
        if Label('//pkg1:c_yes') not in deps or Label('//pkg1:d_no') not in deps:
            fail("deps should be available")
      elif len(ctx.rule.attr.deps.value) != 0:
        fail("deps should be empty")

      return ['deps', 'impl']

  return ['components']

def _aspect_impl(target, ctx):
  res = ['cmdline_aspect on %s' % target.label]

  if hasattr(ctx.rule.attr, 'impl') and type(ctx.rule.attr.impl) != 'DormantDependency':
    fail("impl should be a dormant dependency")

  deps = []
  deps.extend(getattr(ctx.rule.attr, 'deps', []))
  deps.extend(getattr(ctx.rule.attr, 'components', []))

  for dep in deps:
    if AspectInfo in dep:
      res += dep[AspectInfo].res

  return [AspectInfo(res = res)]

cmdline_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = _propagation_attrs,
)
""");

    useConfiguration("--experimental_dormant_deps");
    var analysisResult = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:bin");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly(
            "cmdline_aspect on //pkg1:bin",
            "cmdline_aspect on //pkg1:a_yes",
            "cmdline_aspect on //pkg1:b_no",
            "cmdline_aspect on //pkg1:c_yes",
            "cmdline_aspect on //pkg1:d_no");

    var aspectResult =
        getAspectResult(
            analysisResult.getAspectsMap(), "cmdline_aspect", "//pkg1:bin", "AspectInfo");
    assertThat(aspectResult)
        .containsExactly(
            "cmdline_aspect on @@//pkg1:bin",
            "cmdline_aspect on @@//pkg1:a_yes",
            "cmdline_aspect on @@//pkg1:b_no",
            "cmdline_aspect on @@//pkg1:c_yes",
            "cmdline_aspect on @@//pkg1:d_no");
  }

  @Test
  public void allRuleAttributesAreAvailableInAttrAspectsFunc() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
"""
attr_map = {
  'bool': True,
  'int': 100,
  'int_list': [1, 2, 3],
  'label': Label('//pkg1:dep_1'),
  'label_keyed_string_dict': {Label('//pkg1:dep_1'): 'key1', Label('//pkg1:dep_2'): 'key2'},
  'label_list': [Label('//pkg1:dep_1'), Label('//pkg1:dep_2')],
  'output': Label('//pkg1:output.txt'),
  'output_list': [Label('//pkg1:output_2.txt'), Label('//pkg1:output_3.txt')],
  'string': 'string_value',
  'string_dict': {'key1': 'value1', 'key2': 'value2'},
  'string_keyed_label_dict': {'key1': Label('//pkg1:dep_1'), 'key2': Label('//pkg1:dep_2')},
  'string_list': ['string_value_1', 'string_value_2'],
  'string_list_dict': {'key1': ['string_value_1', 'string_value_2'], 'key2': ['string_value_3', 'string_value_4']},
  'label_list_dict': {'key1': [Label('//pkg1:dep_1')], 'key2': [Label('//pkg1:dep_2')]},
}
def _propagation_attrs(ctx):
  for attr_name, expected_val in attr_map.items():
    if not hasattr(ctx.rule.attr, attr_name):
      fail("'%s' is not an attribute of the rule" % attr_name)

    actual_val = getattr(ctx.rule.attr, attr_name).value
    if expected_val != actual_val:
      fail("'%s' has wrong value: expected %s, got %s" % (attr_name, expected_val, actual_val))
  return []

def _aspect_impl(target, ctx):
  return []

cmdline_aspect = aspect(
  implementation = _aspect_impl,
  attr_aspects = _propagation_attrs,
)

def _rule_impl(ctx):
  if ctx.outputs.output:
    ctx.actions.write(ctx.outputs.output, 'hi')
  if ctx.outputs.output_list:
    ctx.actions.write(ctx.outputs.output_list[0], 'hi')
    ctx.actions.write(ctx.outputs.output_list[1], 'hi')
  return []

my_rule = rule(
  implementation = _rule_impl,
  attrs = {
    "bool": attr.bool(),
    "int": attr.int(),
    "int_list": attr.int_list(),
    "label": attr.label(),
    "label_keyed_string_dict": attr.label_keyed_string_dict(),
    "label_list": attr.label_list(),
    "output": attr.output(),
    "output_list": attr.output_list(),
    "string": attr.string(),
    "string_dict": attr.string_dict(),
    "string_keyed_label_dict": attr.string_keyed_label_dict(),
    "string_list": attr.string_list(),
    "string_list_dict": attr.string_list_dict(),
    "label_list_dict": attr.label_list_dict(),
    })
""");
    scratch.file(
        "pkg1/BUILD",
"""
load("//test:defs.bzl", "my_rule")
my_rule(
    name = 'main_target',
    bool = True,
    int = 100,
    int_list = [1, 2, 3],
    label = ':dep_1',
    label_keyed_string_dict = {'//pkg1:dep_1': 'key1', '//pkg1:dep_2': 'key2'},
    label_list = [':dep_1', ':dep_2'],
    output = 'output.txt',
    output_list = ['output_2.txt', 'output_3.txt'],
    string = 'string_value',
    string_dict = {'key1': 'value1', 'key2': 'value2'},
    string_keyed_label_dict = {'key1': ':dep_1', 'key2': ':dep_2'},
    string_list = ['string_value_1', 'string_value_2'],
    string_list_dict = {'key1': ['string_value_1', 'string_value_2'], 'key2': ['string_value_3', 'string_value_4']},
    label_list_dict = {'key1': [':dep_1'], 'key2': [':dep_2']},
)

my_rule(name = 'dep_1')
my_rule(name = 'dep_2')
""");
    var unused = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:main_target");

    assertThat(getFormattedAspectKeys("cmdline_aspect"))
        .containsExactly("cmdline_aspect on //pkg1:main_target");
  }

  @Test
  public void attributeWithTransition_availableInAttrAspectsFunc() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
"""
def _propagation_attrs(ctx):
  if ctx.rule.label.name != 'main_target':
    return []

  dep = ctx.rule.attr.dep.value
  if not dep or type(dep) != type([]) or dep[0] != Label('//pkg1:dep_1'):
    fail("dep is not available")
  return ['dep']

def _aspect_impl(target, ctx):
  return []

cmdline_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = _propagation_attrs,
)

def _transition_impl(settings, attr):
    return {
      "t1": {"//command_line_option:foo" : "v1"},
      "t2": {"//command_line_option:foo" : "v2"}
    }

simple_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//command_line_option:foo"]
)

def _rule_impl(ctx):
  pass

my_rule = rule(
    implementation = _rule_impl,
    attrs = {
        "dep": attr.label(cfg = simple_transition),
    },
)
""");

    scratch.file(
        "pkg1/BUILD",
"""
load("//test:defs.bzl", "my_rule")
my_rule(name = 'main_target', dep = ':dep_1')
my_rule(name = 'dep_1')
""");
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    useRuleClassProvider(builder.build());

    useConfiguration("--foo=default");
    var unused = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:main_target");

    var aspectKeys =
        getAspectKeys("cmdline_aspect").stream()
            .map(
                k ->
                    k.getLabel()
                        + " with foo = "
                        + k.getConfigurationKey().getOptions().get(DummyTestOptions.class).foo);
    assertThat(aspectKeys)
        .containsExactly(
            "//pkg1:main_target with foo = default",
            "//pkg1:dep_1 with foo = v1",
            "//pkg1:dep_1 with foo = v2");
  }

  private ImmutableList<AspectKey> getAspectKeys(String aspectName) {
    return skyframeExecutor.getEvaluator().getDoneValues().entrySet().stream()
        .filter(
            entry ->
                entry.getKey() instanceof AspectKey
                    && ((AspectKey) entry.getKey())
                        .getAspectClass()
                        .toString()
                        .equals("//test:defs.bzl%" + aspectName))
        .map(e -> (AspectKey) e.getKey())
        .collect(toImmutableList());
  }

  private String formatAspectKey(AspectKey aspectKey) {
    if (aspectKey.getBaseKeys().isEmpty()) {
      return Splitter.on("%").splitToList(aspectKey.getAspectClass().toString()).get(1)
          + " on "
          + aspectKey.getLabel();
    }

    String baseAspects =
        aspectKey.getBaseKeys().stream()
            .map(k -> Splitter.on("%").splitToList(k.getAspectClass().toString()).get(1))
            .collect(joining(","));
    return Splitter.on("%").splitToList(aspectKey.getAspectClass().toString()).get(1)
        + " on "
        + aspectKey.getLabel()
        + " with base aspects: "
        + baseAspects;
  }

  private ImmutableList<String> getFormattedAspectKeys(String aspectName) {
    return skyframeExecutor.getEvaluator().getDoneValues().entrySet().stream()
        .filter(
            entry ->
                entry.getKey() instanceof AspectKey
                    && ((AspectKey) entry.getKey())
                        .getAspectClass()
                        .toString()
                        .equals("//test:defs.bzl%" + aspectName))
        .map(e -> formatAspectKey((AspectKey) e.getKey()))
        .collect(toImmutableList());
  }

  private Sequence<?> getAspectResult(
      Map<AspectKey, ConfiguredAspect> aspectsMap, String aspectName) throws Exception {
    return getAspectResult(aspectsMap, aspectName, null, "AspectInfo");
  }

  private Sequence<?> getAspectResult(
      Map<AspectKey, ConfiguredAspect> aspectsMap,
      String aspectName,
      String targetLabel,
      String providerName)
      throws Exception {
    for (Map.Entry<AspectKey, ConfiguredAspect> entry : aspectsMap.entrySet()) {
      AspectClass aspectClass = entry.getKey().getAspectClass();
      if (aspectClass instanceof StarlarkAspectClass starlarkAspectClass) {
        String aspectExportedName = starlarkAspectClass.getExportedName();
        if (aspectExportedName.equals(aspectName)
            && (targetLabel == null || entry.getKey().getLabel().toString().equals(targetLabel))) {
          return getStarlarkProvider(entry.getValue(), "//test:defs.bzl", providerName)
              .getValue("res", Sequence.class);
        }
      }
    }
    throw new AssertionError("Aspect result not found for aspect: " + aspectName);
  }
}
