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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Sequence;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark aspects propagation functions. */
@RunWith(JUnit4.class)
public final class StarlarkAspectsPropagationFunctionsTest extends AnalysisTestCase {

  private void createTestDefs(String propagationPredicate) throws Exception {
    scratch.file(
        "test/BUILD",
        """
        load('//test:defs.bzl', 'rule_1')
        rule_1(name = 'tool')
        """);
    scratch.file(
        "test/defs.bzl",
        String.format(
            """
            AspectInfo = provider()
            RuleInfo = provider()

            %s

            def _aspect_impl(target, ctx):
              res = ['cmdline_aspect on ' + str(target.label)]
              for dep in ctx.rule.attr.deps:
                if AspectInfo in dep:
                  res.extend(dep[AspectInfo].res)
              return [AspectInfo(res = res)]

            cmdline_aspect = aspect(
              implementation = _aspect_impl,
              propagation_predicate = _propagation_predicate,
              attrs = {
                'allowed_pkg': attr.string(default = '*'),
                'ignored_tag': attr.string(default = ''),
                'allowed_rule_file': attr.string(default = '*'),
                'allowed_rule_name': attr.string(default = '*'),
                '_tool': attr.label(default = Label('//test:tool'))},
              attr_aspects = ['deps'],
            )

            aspect_with_required_provider = aspect(
                implementation = _aspect_impl,
                required_providers = [RuleInfo],
                propagation_predicate = _propagation_predicate,
                attr_aspects = ['deps'],
            )

            def _rule_impl(ctx):
              pass

            rule_1 = rule(
              implementation = _rule_impl,
              attrs = {'deps': attr.label_list()},
            )

            rule_2 = rule(
              implementation = _rule_impl,
              attrs = {'deps': attr.label_list()},
            )

            def _out_rule_impl(ctx):
              if ctx.outputs.out:
                ctx.actions.write(ctx.outputs.out, 'hi')
              return []

            out_rule = rule(
              implementation = _out_rule_impl,
              attrs = {'deps': attr.label_list(), 'out': attr.output()},
            )

            def _rule_with_provider_impl(ctx):
              return [RuleInfo()]

            rule_with_provider = rule(
              implementation = _rule_with_provider_impl,
              attrs = {'deps': attr.label_list()},
              provides = [RuleInfo],
            )

            aspect_on_rule = aspect(
                implementation = _aspect_impl,
                propagation_predicate = _propagation_predicate,
                attr_aspects = ['deps'],
            )

            rule_with_aspect_on_deps = rule(
                implementation = _rule_impl,
                attrs = {'deps': attr.label_list(aspects = [aspect_on_rule])},
            )
            """,
            propagationPredicate));
  }

  private void createTestPackages() throws Exception {
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'rule_1', 'rule_2', 'out_rule', 'rule_with_aspect_on_deps')
        rule_1(name = 't1', deps = [':t2', '//pkg2:t2'], tags = ['no-aspect'])
        rule_2(name = 't2', tags = ['another-tag'])
        out_rule(name = 'target_with_output', out = 'my_out.txt', deps = [':t2'])
        rule_with_aspect_on_deps(name = 'target_with_aspect_on_deps', deps = [':t2', '//pkg2:t2'])
        alias(name = 'alias_1', actual = ':alias_2')
        alias(name = 'alias_2', actual = ':actual')
        rule_1(name = 'actual')
        """);
    scratch.file(
        "pkg2/BUILD",
        """
        load('//test:defs.bzl', 'rule_1', 'rule_2', 'rule_with_provider')
        rule_2(name = 't1', deps = [':t2', '//pkg1:t2'])
        rule_1(name = 't2', tags = ['no-aspect', 'another-tag'])
        rule_with_provider(name = 'target_with_provider', deps = [':t2'])
        """);
  }

  @Test
  public void propagationPredicateOnTargetPackage_aspectPropagatedToSatisfyingTargets()
      throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          if ctx.attr.allowed_pkg != '*' and ctx.rule.label.package != ctx.attr.allowed_pkg:
            return False
          return True
        """);
    createTestPackages();

    var analysisResult =
        update(
            ImmutableList.of("//test:defs.bzl%cmdline_aspect"),
            ImmutableMap.of("allowed_pkg", "pkg1"),
            "//pkg1:t1",
            "//pkg2:t1");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%cmdline_aspect");
    // Only the keys to the targets that satisfy the aspect's propagation predicate are present.
    assertThat(aspectKeys)
        .containsExactly("cmdline_aspect on //pkg1:t1", "cmdline_aspect on //pkg1:t2");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly("cmdline_aspect on @@//pkg1:t1", "cmdline_aspect on @@//pkg1:t2");
  }

  @Test
  public void propagationPredicateOnTargetTags_aspectPropagatedToSatisfyingTargets()
      throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          if ctx.attr.ignored_tag != '' and ctx.attr.ignored_tag in ctx.rule.attr.tags.value:
            return False
          return True
        """);
    createTestPackages();

    var analysisResult =
        update(
            ImmutableList.of("//test:defs.bzl%cmdline_aspect"),
            ImmutableMap.of("ignored_tag", "no-aspect"),
            "//pkg1:t1",
            "//pkg2:t1");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%cmdline_aspect");
    // Only the keys to the targets that satisfy the aspect's propagation predicate are present.
    assertThat(aspectKeys)
        .containsExactly("cmdline_aspect on //pkg2:t1", "cmdline_aspect on //pkg1:t2");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly("cmdline_aspect on @@//pkg2:t1", "cmdline_aspect on @@//pkg1:t2");
  }

  @Test
  public void propagationPredicateOnRuleKind_aspectPropagatedToSatisfyingTargets()
      throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          qualified_kind = ctx.rule.qualified_kind
          allowed_rule_file = ctx.attr.allowed_rule_file
          allowed_rule_name = ctx.attr.allowed_rule_name

          if allowed_rule_file != '*' and qualified_kind.file_label != Label(allowed_rule_file):
            return False

          if allowed_rule_name != '*' and qualified_kind.rule_name != allowed_rule_name:
            return False

          return True
        """);
    createTestPackages();

    var analysisResult =
        update(
            ImmutableList.of("//test:defs.bzl%cmdline_aspect"),
            ImmutableMap.of(
                "allowed_rule_file", "@@//test:defs.bzl", "allowed_rule_name", "rule_1"),
            "//pkg1:t1",
            "//pkg2:t1");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%cmdline_aspect");
    // Only the keys to the targets that satisfy the aspect's propagation predicate are present.
    assertThat(aspectKeys)
        .containsExactly("cmdline_aspect on //pkg1:t1", "cmdline_aspect on //pkg2:t2");

    var aspectResult = getAspectResult(analysisResult.getAspectsMap(), "cmdline_aspect");
    assertThat(aspectResult)
        .containsExactly("cmdline_aspect on @@//pkg1:t1", "cmdline_aspect on @@//pkg2:t2");
  }

  @Test
  public void accessPrivateAspectAttribute_fails() throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          tool = ctx.attr._tool
          return True
        """);
    createTestPackages();

    reporter.removeHandler(failFastHandler);
    assertThrows(
        ViewCreationFailedException.class,
        () -> update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:t1"));
    assertContainsEvent("'_tool' is not a public parameter of the aspect.");
  }

  @Test
  public void aspectOnOutputFile_propagationPredicateNotUsed() throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          return False
        """);
    createTestPackages();

    var unused = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:my_out.txt");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%cmdline_aspect");
    // The propagation predicate is not used for output files, so the aspect key is created even if
    // the propagation predicate is not satisfied.
    assertThat(aspectKeys).containsExactly("cmdline_aspect on //pkg1:my_out.txt");
  }

  @Test
  public void aspectPropagatedFromRule_propagationPredicateIsUsed() throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          if ctx.rule.label == Label('//pkg1:t2'):
            return False
          return True
        """);
    createTestPackages();

    var unused = update("//pkg1:target_with_aspect_on_deps");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_on_rule");
    // The propagation predicate is used for the aspect_on_rule, so the aspect key is not created
    // for //pkg1:t2.
    assertThat(aspectKeys).containsExactly("aspect_on_rule on //pkg2:t2");
  }

  @Test
  public void aspectOnAlias_propagationPredicateRunOnActualTarget() throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          if ctx.rule.qualified_kind.rule_name == 'rule_1':
            return True
          return False
        """);
    createTestPackages();

    var unused = update(ImmutableList.of("//test:defs.bzl%cmdline_aspect"), "//pkg1:alias_1");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%cmdline_aspect");
    // The propagation predicate is evaluated on the actual target of the alias which in this case
    // satisfies the predicate.
    assertThat(aspectKeys)
        .containsExactly(
            "cmdline_aspect on //pkg1:alias_1",
            "cmdline_aspect on //pkg1:alias_2",
            "cmdline_aspect on //pkg1:actual");
  }

  @Test
  public void requiredProviderSatisfied_propagationPredicateNotSatisfied_aspectNotPropagated()
      throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          return False
        """);
    createTestPackages();

    var unused =
        update(
            ImmutableList.of("//test:defs.bzl%aspect_with_required_provider"),
            "//pkg2:target_with_provider");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_with_required_provider");
    // The propagation predicate is not satisfied, so the aspect is not propagated even though its
    // required provider is satisfied.
    assertThat(aspectKeys).isEmpty();
  }

  @Test
  public void requiredProviderSatisfied_propagationPredicateSatisfied_aspectPropagated()
      throws Exception {
    createTestDefs(
        """
        def _propagation_predicate(ctx):
          return True
        """);
    createTestPackages();

    var unused =
        update(
            ImmutableList.of("//test:defs.bzl%aspect_with_required_provider"),
            "//pkg2:target_with_provider");

    var aspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_with_required_provider");
    // The propagation predicate and the required provider are satisfied, so the aspect is
    // propagated to the target.
    assertThat(aspectKeys)
        .containsExactly("aspect_with_required_provider on //pkg2:target_with_provider");
  }

  @Test
  public void aspectOnAspect_eachPropagationPredicateEvaluatedSeparately() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        CInfo = provider()
        BInfo = provider()
        AInfo = provider()

        def _aspect_a_propagation_predicate(ctx):
          return True

        def _aspect_b_propagation_predicate(ctx):
          if ctx.rule.label.package == 'pkg1':
            return True
          return False

        def _aspect_c_propagation_predicate(ctx):
          return True

        def _aspect_a_impl(target, ctx):
          res = 'aspect_a on %s' % target.label
          if BInfo in target:
            res += ' with BInfo'
          if CInfo in target:
            res += ' with CInfo'

          res_list = [res]

          for dep in ctx.rule.attr.deps:
            if AInfo in dep:
              res_list.extend(dep[AInfo].res)

          return [AInfo(res = res_list)]

        def _aspect_b_impl(target, ctx):
          return [BInfo()]

        def _aspect_c_impl(target, ctx):
          return [CInfo()]

        aspect_a = aspect(
          implementation = _aspect_a_impl,
          propagation_predicate = _aspect_a_propagation_predicate,
          attr_aspects = ['deps'],
          required_aspect_providers = [[BInfo], [CInfo]],
        )

        aspect_b = aspect(
            implementation = _aspect_b_impl,
            propagation_predicate = _aspect_b_propagation_predicate,
            attr_aspects = ['deps'],
            provides = [BInfo],
        )

        aspect_c = aspect(
            implementation = _aspect_c_impl,
            propagation_predicate = _aspect_c_propagation_predicate,
            attr_aspects = ['deps'],
            provides = [CInfo],
        )

        def _rule_impl(ctx):
          pass

        my_rule = rule(
          implementation = _rule_impl,
          attrs = {'deps': attr.label_list()},
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 't1', deps = [':t2', '//pkg2:t2'])
        my_rule(name = 't2')
        """);
    scratch.file(
        "pkg2/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 't1', deps = [':t2'])
        my_rule(name = 't2')
        """);

    var analysisResult =
        update(
            ImmutableList.of(
                "//test:defs.bzl%aspect_c", "//test:defs.bzl%aspect_b", "//test:defs.bzl%aspect_a"),
            "//pkg1:t1",
            "//pkg2:t1");

    // The propagation predicate of each aspect is evaluated separately, then the aspect-on-aspect
    // relation is created between the filtered aspects.
    var aAspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_a");
    assertThat(aAspectKeys)
        .containsExactly(
            "aspect_a on //pkg1:t1 with base aspects: aspect_b,aspect_c",
            "aspect_a on //pkg1:t2 with base aspects: aspect_b,aspect_c",
            "aspect_a on //pkg2:t1 with base aspects: aspect_c",
            "aspect_a on //pkg2:t2 with base aspects: aspect_c");

    var bAspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_b");
    assertThat(bAspectKeys).containsExactly("aspect_b on //pkg1:t1", "aspect_b on //pkg1:t2");

    var cAspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_c");
    assertThat(cAspectKeys)
        .containsExactly(
            "aspect_c on //pkg1:t1",
            "aspect_c on //pkg1:t2",
            "aspect_c on //pkg2:t1",
            "aspect_c on //pkg2:t2");

    var aspectAonPkg1T1 =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg1:t1", "AInfo");
    assertThat(aspectAonPkg1T1)
        .containsExactly(
            "aspect_a on @@//pkg1:t1 with BInfo with CInfo",
            "aspect_a on @@//pkg1:t2 with BInfo with CInfo",
            "aspect_a on @@//pkg2:t2 with CInfo");

    var aspectAonPkg2T1 =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg2:t1", "AInfo");
    assertThat(aspectAonPkg2T1)
        .containsExactly(
            "aspect_a on @@//pkg2:t1 with CInfo", "aspect_a on @@//pkg2:t2 with CInfo");
  }

  @Test
  public void requiredAspects_propagationPredicateOfRequiredAspectIsUsed() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/defs.bzl",
        """
        AInfo = provider()
        BInfo = provider()

        def _aspect_a_propagation_predicate(ctx):
          return True

        def _aspect_b_propagation_predicate(ctx):
          if ctx.rule.label.package == 'pkg1':
            return True
          return False

        def _aspect_a_impl(target, ctx):
          res = 'aspect_a on %s' % target.label
          if BInfo in target:
            res += ' with BInfo'
          else:
            res += ' without BInfo'

          res_list = [res]

          for dep in ctx.rule.attr.deps:
            if AInfo in dep:
              res_list.extend(dep[AInfo].res)

          return [AInfo(res = res_list)]

        def _aspect_b_impl(target, ctx):
          return [BInfo()]

        aspect_b = aspect(
            implementation = _aspect_b_impl,
            propagation_predicate = _aspect_b_propagation_predicate,
        )

        aspect_a = aspect(
          implementation = _aspect_a_impl,
          propagation_predicate = _aspect_a_propagation_predicate,
          attr_aspects = ['deps'],
          requires = [aspect_b],
        )

        def _rule_impl(ctx):
          pass

        my_rule = rule(
          implementation = _rule_impl,
          attrs = {'deps': attr.label_list()},
        )
        """);
    scratch.file(
        "pkg1/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 't1', deps = [':t2', '//pkg2:t2'])
        my_rule(name = 't2')
        """);
    scratch.file(
        "pkg2/BUILD",
        """
        load('//test:defs.bzl', 'my_rule')
        my_rule(name = 't1', deps = [':t2'])
        my_rule(name = 't2')
        """);

    var analysisResult =
        update(ImmutableList.of("//test:defs.bzl%aspect_a"), "//pkg1:t1", "//pkg2:t1");

    // The propagation predicate of both aspects is used.
    var aAspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_a");
    assertThat(aAspectKeys)
        .containsExactly(
            "aspect_a on //pkg1:t1 with base aspects: aspect_b",
            "aspect_a on //pkg1:t2 with base aspects: aspect_b",
            "aspect_a on //pkg2:t1",
            "aspect_a on //pkg2:t2");

    var bAspectKeys = getFormattedAspectKeys("//test:defs.bzl%aspect_b");
    assertThat(bAspectKeys).containsExactly("aspect_b on //pkg1:t1", "aspect_b on //pkg1:t2");

    var aspectAonPkg1T1 =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg1:t1", "AInfo");
    assertThat(aspectAonPkg1T1)
        .containsExactly(
            "aspect_a on @@//pkg1:t1 with BInfo",
            "aspect_a on @@//pkg1:t2 with BInfo",
            "aspect_a on @@//pkg2:t2 without BInfo");
    var aspectAonPkg2T1 =
        getAspectResult(analysisResult.getAspectsMap(), "aspect_a", "//pkg2:t1", "AInfo");
    assertThat(aspectAonPkg2T1)
        .containsExactly(
            "aspect_a on @@//pkg2:t1 without BInfo", "aspect_a on @@//pkg2:t2 without BInfo");
  }

  private String formatAspectKey(AspectKey aspectKey) {
    if (aspectKey.getBaseKeys().isEmpty()) {
      return Splitter.on("%").splitToList(aspectKey.getAspectClass().toString()).get(1)
          + " on "
          + aspectKey.getLabel();
    }

    String baseAspects =
        String.join(
            ",",
            aspectKey.getBaseKeys().stream()
                .map(k -> Splitter.on("%").splitToList(k.getAspectClass().toString()).get(1))
                .collect(toImmutableList()));
    return Splitter.on("%").splitToList(aspectKey.getAspectClass().toString()).get(1)
        + " on "
        + aspectKey.getLabel()
        + " with base aspects: "
        + baseAspects;
  }

  private ImmutableList<String> getFormattedAspectKeys(String aspectLabel) {
    return skyframeExecutor.getEvaluator().getDoneValues().entrySet().stream()
        .filter(
            entry ->
                entry.getKey() instanceof AspectKey
                    && ((AspectKey) entry.getKey()).getAspectClass().toString().equals(aspectLabel))
        .map(e -> formatAspectKey((AspectKey) e.getKey()))
        .collect(toImmutableList());
  }

  private Sequence<?> getAspectResult(
      Map<AspectKey, ConfiguredAspect> aspectsMap, String aspectName) throws Exception {
    return getAspectResult(aspectsMap, aspectName, null, "AspectInfo");
  }

  @Nullable
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
    return null;
  }
}
