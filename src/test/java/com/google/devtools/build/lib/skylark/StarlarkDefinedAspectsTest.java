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
package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.analysis.OutputGroupInfo.INTERNAL_SUFFIX;
import static java.util.stream.Collectors.toList;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcProtoProvider;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark aspects */
@RunWith(JUnit4.class)
public class StarlarkDefinedAspectsTest extends AnalysisTestCase {
  protected boolean keepGoing() {
    return false;
  }

  private static final String LINE_SEPARATOR = System.lineSeparator();

  @Before
  public final void initializeToolsConfigMock() throws Exception {
    // Required for tests including the objc_library rule.
    MockObjcSupport.setup(mockToolsConfig);
    // Required for tests including the proto_library rule.
    MockProtoSupport.setup(mockToolsConfig);
  }

  @Test
  public void simpleAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
  }

  @Test
  public void aspectWithSingleDeclaredProvider() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "def _impl(target, ctx):",
        "   return foo()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());

    StarlarkProvider.Key fooKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "foo");

    assertThat(configuredAspect.get(fooKey).getProvider().getKey()).isEqualTo(fooKey);
  }

  @Test
  public void aspectWithDeclaredProviders() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "bar = provider()",
        "def _impl(target, ctx):",
        "   return [foo(), bar()]",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());

    StarlarkProvider.Key fooKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "foo");
    StarlarkProvider.Key barKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "bar");

    assertThat(configuredAspect.get(fooKey).getProvider().getKey()).isEqualTo(fooKey);
    assertThat(configuredAspect.get(barKey).getProvider().getKey()).isEqualTo(barKey);
  }

  @Test
  public void aspectWithDeclaredProvidersInAStruct() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "foo = provider()",
        "bar = provider()",
        "def _impl(target, ctx):",
        "   return struct(foobar='foobar', providers=[foo(), bar()])",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());

    StarlarkProvider.Key fooKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "foo");
    StarlarkProvider.Key barKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "bar");

    assertThat(configuredAspect.get(fooKey).getProvider().getKey()).isEqualTo(fooKey);
    assertThat(configuredAspect.get(barKey).getProvider().getKey()).isEqualTo(barKey);
  }

  private Iterable<String> getAspectDescriptions(AnalysisResult analysisResult) {
    return Iterables.transform(
        analysisResult.getAspectsMap().keySet(),
        aspectKey ->
            String.format("%s(%s)", aspectKey.getAspectClass().getName(), aspectKey.getLabel()));
  }

  @Test
  public void aspectCommandLineLabel() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("//test:aspect.bzl%MyAspect(//test:xxx)");
  }

  @Test
  public void aspectCommandLineRepoLabel() throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        scratch.readFile("WORKSPACE"),
        "local_repository(name='local', path='local/repo')");
    scratch.file("local/repo/WORKSPACE");
    scratch.file(
        "local/repo/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("local/repo/BUILD");

    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("@local//:aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    assertThat(getAspectDescriptions(analysisResult))
        .containsExactly("@local//:aspect.bzl%MyAspect(//test:xxx)");
  }

  private Iterable<String> getLabelsToBuild(AnalysisResult analysisResult) {
    return Iterables.transform(
        analysisResult.getTargetsToBuild(),
        configuredTarget -> configuredTarget.getLabel().toString());
  }

  @Test
  public void aspectAllowsFragmentsToBeSpecified() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl, fragments=['java'], host_fragments=['cpp'])");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");

    AspectKey key = Iterables.getOnlyElement(analysisResult.getAspectsMap().keySet());
    AspectValue aspectValue =
        (AspectValue) skyframeExecutor.getEvaluatorForTesting().getExistingValue(key);
    AspectDefinition aspectDefinition = aspectValue.getAspect().getDefinition();
    assertThat(
            aspectDefinition
                .getConfigurationFragmentPolicy()
                .isLegalConfigurationFragment(JavaConfiguration.class, NoTransition.INSTANCE))
        .isTrue();
    assertThat(
            aspectDefinition
                .getConfigurationFragmentPolicy()
                .isLegalConfigurationFragment(JavaConfiguration.class, HostTransition.INSTANCE))
        .isFalse();
    assertThat(
            aspectDefinition
                .getConfigurationFragmentPolicy()
                .isLegalConfigurationFragment(CppConfiguration.class, NoTransition.INSTANCE))
        .isFalse();
    assertThat(
            aspectDefinition
                .getConfigurationFragmentPolicy()
                .isLegalConfigurationFragment(CppConfiguration.class, HostTransition.INSTANCE))
        .isTrue();
  }

  @Test
  public void aspectPropagating() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = depset([target.label], transitive = [i.target_labels for i in ctx.rule.attr.deps])",
        "   c = depset([ctx.rule.kind], transitive = [i.rule_kinds for i in ctx.rule.attr.deps])",
        "   return struct(target_labels = s, rule_kinds = c)",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        "     deps = [':yyy'],",
        ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    assertThat(configuredAspect).isNotNull();
    Object names = configuredAspect.get("target_labels");
    assertThat(names).isInstanceOf(Depset.class);
    assertThat(
            Iterables.transform(
                ((Depset) names).toList(),
                o -> {
                  assertThat(o).isInstanceOf(Label.class);
                  return o.toString();
                }))
        .containsExactly("//test:xxx", "//test:yyy");
    Object ruleKinds = configuredAspect.get("rule_kinds");
    assertThat(ruleKinds).isInstanceOf(Depset.class);
    assertThat(((Depset) ruleKinds).toList()).containsExactly("java_library");
  }

  @Test
  public void aspectsPropagatingForDefaultAndImplicit() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = []",
        "   c = []",
        "   a = ctx.rule.attr",
        "   if getattr(a, '_defaultattr', None):",
        "       s += [a._defaultattr.target_labels]",
        "       c += [a._defaultattr.rule_kinds]",
        "   if getattr(a, '_cc_toolchain', None):",
        "       s += [a._cc_toolchain.target_labels]",
        "       c += [a._cc_toolchain.rule_kinds]",
        "   return struct(",
        "       target_labels = depset([target.label], transitive = s),",
        "       rule_kinds = depset([ctx.rule.kind], transitive = c))",
        "",
        "def _rule_impl(ctx):",
        "   pass",
        "",
        "my_rule = rule(implementation = _rule_impl,",
        "   attrs = { '_defaultattr' : attr.label(default = Label('//test:xxx')) },",
        ")",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['_defaultattr', '_cc_toolchain'],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "cc_library(",
        "     name = 'xxx',",
        ")",
        "my_rule(",
        "     name = 'yyy',",
        ")");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:yyy");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    assertThat(configuredAspect).isNotNull();
    Object nameSet = configuredAspect.get("target_labels");
    ImmutableList<String> names =
        ImmutableList.copyOf(
            Iterables.transform(
                ((Depset) nameSet).toList(),
                o -> {
                  assertThat(o).isInstanceOf(Label.class);
                  return ((Label) o).getName();
                }));

    assertThat(names).containsAtLeast("xxx", "yyy");
    // Third is the C++ toolchain; its name changes between Blaze and Bazel.
    assertThat(names).hasSize(3);
  }

  @Test
  public void aspectsDirOnMergedTargets() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct(aspect_provider = 'data')",
        "",
        "p = provider()",
        "MyAspect = aspect(implementation=_impl)",
        "def _rule_impl(ctx):",
        "   if ctx.attr.dep:",
        "      return [p(dir = dir(ctx.attr.dep))]",
        "   return [p()]",
        "",
        "my_rule = rule(implementation = _rule_impl,",
        "   attrs = { 'dep' : attr.label(aspects = [MyAspect]) },",
        ")");
    StarlarkProvider.Key providerKey =
        new StarlarkProvider.Key(Label.parseAbsoluteUnchecked("//test:aspect.bzl"), "p");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx',)",
        "my_rule(name = 'yyy', dep = ':xxx')");
    AnalysisResult analysisResult = update("//test:yyy");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());

    StructImpl names = (StructImpl) target.get(providerKey);
    assertThat((Iterable<?>) names.getValue("dir"))
        .containsExactly(
            "actions",
            "aspect_provider",
            "data_runfiles",
            "default_runfiles",
            "files",
            "files_to_run",
            "label",
            "output_groups");
  }

  @Test
  public void aspectWithOutputGroupsDeclaredProvider() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   f = target[OutputGroupInfo]._hidden_top_level" + INTERNAL_SUFFIX,
        "   return [OutputGroupInfo(my_result = f)]",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        ")");
    scratch.file(
        "test/BUILD", "java_library(", "     name = 'xxx',", "     srcs = ['A.java'],", ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    OutputGroupInfo outputGroupInfo = OutputGroupInfo.get(configuredAspect);

    assertThat(outputGroupInfo).isNotNull();
    NestedSet<Artifact> names = outputGroupInfo.getOutputGroup("my_result");
    assertThat(names.toList()).isNotEmpty();
    NestedSet<Artifact> expectedSet =
        OutputGroupInfo.get(getConfiguredTarget("//test:xxx"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    assertThat(names.toList()).containsExactlyElementsIn(expectedSet.toList());
  }

  @Test
  public void aspectWithOutputGroupsAsListDeclaredProvider() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   g = target[OutputGroupInfo]._hidden_top_level" + INTERNAL_SUFFIX,
        "   return [OutputGroupInfo(my_result=g.to_list())]",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        ")");
    scratch.file(
        "test/BUILD", "java_library(", "     name = 'xxx',", "     srcs = ['A.java'],", ")");

    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(
            Iterables.transform(
                analysisResult.getTargetsToBuild(),
                configuredTarget -> configuredTarget.getLabel().toString()))
        .containsExactly("//test:xxx");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    OutputGroupInfo outputGroupInfo = OutputGroupInfo.get(configuredAspect);
    assertThat(outputGroupInfo).isNotNull();
    NestedSet<Artifact> names = outputGroupInfo.getOutputGroup("my_result");
    assertThat(names.toList()).isNotEmpty();
    NestedSet<Artifact> expectedSet =
        OutputGroupInfo.get(getConfiguredTarget("//test:xxx"))
            .getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL);
    assertThat(names.toList()).containsExactlyElementsIn(expectedSet.toList());
  }

  @Test
  public void aspectsFromStarlarkRules() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   s = depset([target.label], transitive = [i.target_labels for i in ctx.rule.attr.deps])",
        "   return struct(target_labels = s)",
        "",
        "def _rule_impl(ctx):",
        "   s = depset(transitive = [i.target_labels for i in ctx.attr.attr])",
        "   return struct(rule_deps = s)",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        "   attr_aspects=['deps'],",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label_list(mandatory=True, allow_files=True, aspects = [MyAspect]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = [':yyy'],",
        ")");

    AnalysisResult analysisResult = update("//test:xxx");
    assertThat(getLabelsToBuild(analysisResult)).containsExactly("//test:xxx");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    Object names = target.get("rule_deps");
    assertThat(names).isInstanceOf(Depset.class);
    assertThat(
            Iterables.transform(
                ((Depset) names).toList(),
                o -> {
                  assertThat(o).isInstanceOf(Label.class);
                  return o.toString();
                }))
        .containsExactly("//test:yyy");
  }

  @Test
  public void aspectsNonExported() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return []",
        "",
        "def _rule_impl(ctx):",
        "   pass",
        "",
        "def mk_aspect():",
        "   return aspect(implementation=_aspect_impl)",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : attr.label_list(aspects = [mk_aspect()]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = [':yyy'],",
        ")");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult analysisResult = update("//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(analysisResult.hasError()).isTrue();
    } catch (ViewCreationFailedException | TargetParsingException e) {
      // expected
    }

    assertContainsEvent("ERROR /workspace/test/aspect.bzl:11:38");
    assertContainsEvent("Aspects should be top-level values in extension files that define them.");
  }

  @Test
  public void providerNonExported() throws Exception {
    scratch.file(
        "test/rule.bzl",
        "def mk_provider():",
        "   return provider()",
        "def _rule_impl(ctx):",
        "   pass",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : attr.label_list(providers = [mk_provider()]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:rule.bzl', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = [':yyy'],",
        ")");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult analysisResult = update("//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(analysisResult.hasError()).isTrue();
    } catch (ViewCreationFailedException | TargetParsingException e) {
      // expected
    }

    assertContainsEvent("ERROR /workspace/test/rule.bzl:7:38");
    assertContainsEvent(
        "Providers should be top-level values in extension files that define them.");
  }

  @Test
  public void aspectOnLabelAttr() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct(aspect_data='foo')",
        "",
        "def _rule_impl(ctx):",
        "   return struct(data=ctx.attr.attr.aspect_data)",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label(aspects = [MyAspect]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = ':yyy',",
        ")");

    AnalysisResult analysisResult = update("//test:xxx");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    Object value = target.get("data");
    assertThat(value).isEqualTo("foo");
  }

  @Test
  public void labelKeyedStringDictAllowsAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct(aspect_data=target.label.name)",
        "",
        "def _rule_impl(ctx):",
        "   return struct(",
        "       data=','.join(['{}:{}'.format(dep.aspect_data, val)",
        "                      for dep, val in ctx.attr.attr.items()]))",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label_keyed_string_dict(aspects = [MyAspect]) },",
        ")");

    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "java_library(",
        "     name = 'yyy',",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = {':yyy': 'zzz'},",
        ")");

    AnalysisResult analysisResult = update("//test:xxx");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    Object value = target.get("data");
    assertThat(value).isEqualTo("yyy:zzz");
  }

  @Test
  public void aspectsDoNotAttachToFiles() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"), "bind(name = 'yyy', actual = '//test:zzz.jar')");
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
        ")");
    scratch.file("test/zzz.jar");
    scratch.file(
        "test/BUILD",
        "exports_files(['zzz.jar'])",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        "     deps = ['//external:yyy'],",
        ")");

    AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void aspectsDoNotAttachToTopLevelFiles() throws Exception {
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"), "bind(name = 'yyy', actual = '//test:zzz.jar')");
    scratch.file(
        "test/aspect.bzl",
        "p = provider()",
        "def _impl(target, ctx):",
        "   return [p()]",
        "",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attr_aspects=['deps'],",
        ")");
    scratch.file("test/zzz.jar");
    scratch.file(
        "test/BUILD",
        "exports_files(['zzz.jar'])",
        "java_library(",
        "     name = 'xxx',",
        "     srcs = ['A.java'],",
        "     deps = ['//external:yyy'],",
        ")");

    AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:zzz.jar");
    assertThat(result.hasError()).isFalse();
    assertThat(
            Iterables.getOnlyElement(result.getAspectsMap().values())
                .getProviders()
                .getProviderCount())
        .isEqualTo(0);
  }

  @Test
  public void aspectFailingExecution() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return 1 // 0",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:13: in "
            + "//test:aspect.bzl%MyAspect aspect on java_library rule //test:xxx: \n"
            + "Traceback (most recent call last):"
            + LINE_SEPARATOR
            + "\tFile \"/workspace/test/BUILD\", line 1"
            + LINE_SEPARATOR
            + "\t\t//test:aspect.bzl%MyAspect(...)"
            + LINE_SEPARATOR
            + "\tFile \"/workspace/test/aspect.bzl\", line 2, in _impl"
            + LINE_SEPARATOR
            + "\t\t1 // 0"
            + LINE_SEPARATOR
            + "integer division by zero");
  }

  @Test
  public void aspectFailingReturnsNotAStruct() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return 0",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Aspect implementation should return a struct, a list, or a provider "
            + "instance, but got int");
  }

  @Test
  public void aspectFailingOrphanArtifacts() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  ctx.actions.declare_file('missing_in_action.txt')",
        "  return struct()",
        "",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:1:13: in "
            + "//test:aspect.bzl%MyAspect aspect on java_library rule //test:xxx: \n"
            + "\n"
            + "\n"
            + "The following files have no generating action:\n"
            + "test/missing_in_action.txt\n");
  }

  @Test
  public void aspectSkippingOrphanArtifactsWithLocation() throws Exception {
    scratch.file(
        "simple/print.bzl",
        "def _print_expanded_location_impl(target, ctx):",
        "    return struct(result=ctx.expand_location(ctx.rule.attr.cmd, []))",
        "",
        "print_expanded_location = aspect(",
        "    implementation = _print_expanded_location_impl,",
        ")");
    scratch.file(
        "simple/BUILD",
        "filegroup(",
        "    name = \"files\",",
        "    srcs = [\"afile\"],",
        ")",
        "",
        "genrule(",
        "    name = \"concat_all_files\",",
        "    srcs = [\":files\"],",
        "    outs = [\"concatenated.txt\"],",
        "    cmd = \"$(location :files)\"",
        ")");

    reporter.removeHandler(failFastHandler);
    AnalysisResult analysisResult =
        update(
            ImmutableList.of("//simple:print.bzl%print_expanded_location"),
            "//simple:concat_all_files");
    assertThat(analysisResult.hasError()).isFalse();
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    String result = (String) configuredAspect.get("result");

    assertThat(result).isEqualTo("simple/afile");
  }

  @Test
  public void topLevelAspectIsNotAnAspect() throws Exception {
    scratch.file("test/aspect.bzl", "MyAspect = 4");
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("MyAspect from //test:aspect.bzl is not an aspect");
  }

  @Test
  public void duplicateOutputGroups() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  f = ctx.actions.declare_file('f.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return struct(output_groups = { 'duplicate' : depset([f]) })",
        "",
        "MyAspect = aspect(implementation=_impl)",
        "def _rule_impl(ctx):",
        "  g = ctx.actions.declare_file('g.txt')",
        "  ctx.actions.write(g, 'g')",
        "  return struct(output_groups = { 'duplicate' : depset([g]) })",
        "my_rule = rule(_rule_impl)",
        "def _noop(ctx):",
        "  pass",
        "rbase = rule(_noop, attrs = { 'dep' : attr.label(aspects = [MyAspect]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule', 'rbase')",
        "my_rule(name = 'xxx')",
        "rbase(name = 'yyy', dep = ':xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update("//test:yyy");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:6: Output group duplicate provided twice");
  }

  @Test
  public void outputGroupsFromOneAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.actions.declare_file(target.label.name + '_a1.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return struct(output_groups = { 'a1_group' : depset([f]) })",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = {k:ctx.attr.dep.output_groups[k] for k in ctx.attr.dep.output_groups}",
        "  return struct(output_groups = og)",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')");

    AnalysisResult analysisResult = update("//test:xxx");
    OutputGroupInfo outputGroupInfo =
        OutputGroupInfo.get(Iterables.getOnlyElement(analysisResult.getTargetsToBuild()));
    assertThat(getOutputGroupContents(outputGroupInfo, "a1_group"))
        .containsExactly("test/base_a1.txt");
  }

  @Test
  public void outputGroupsDeclaredProviderFromOneAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.actions.declare_file(target.label.name + '_a1.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return [OutputGroupInfo(a1_group = depset([f]))]",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  return [OutputGroupInfo(a1_group = ctx.attr.dep[OutputGroupInfo].a1_group)]",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')");

    AnalysisResult analysisResult = update("//test:xxx");
    OutputGroupInfo outputGroupInfo =
        OutputGroupInfo.get(Iterables.getOnlyElement(analysisResult.getTargetsToBuild()));
    assertThat(getOutputGroupContents(outputGroupInfo, "a1_group"))
        .containsExactly("test/base_a1.txt");
  }

  @Test
  public void outputGroupsFromTwoAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.actions.declare_file(target.label.name + '_a1.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return struct(output_groups = { 'a1_group' : depset([f]) })",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = {k:ctx.attr.dep.output_groups[k] for k in ctx.attr.dep.output_groups}",
        "  return struct(output_groups = og)",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })",
        "def _a2_impl(target, ctx):",
        "  g = ctx.actions.declare_file(target.label.name + '_a2.txt')",
        "  ctx.actions.write(g, 'f')",
        "  return struct(output_groups = { 'a2_group' : depset([g]) })",
        "",
        "a2 = aspect(implementation=_a2_impl, attr_aspects = ['dep'])",
        "my_rule2 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a2]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1', 'my_rule2')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')",
        "my_rule2(name = 'yyy', dep = ':xxx')");

    AnalysisResult analysisResult = update("//test:yyy");
    OutputGroupInfo outputGroupInfo =
        OutputGroupInfo.get(Iterables.getOnlyElement(analysisResult.getTargetsToBuild()));
    assertThat(getOutputGroupContents(outputGroupInfo, "a1_group"))
        .containsExactly("test/base_a1.txt");
    assertThat(getOutputGroupContents(outputGroupInfo, "a2_group"))
        .containsExactly("test/xxx_a2.txt");
  }

  @Test
  public void outputGroupsDeclaredProvidersFromTwoAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.actions.declare_file(target.label.name + '_a1.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return [OutputGroupInfo(a1_group = depset([f]))]",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = dict()",
        "  dep_og = ctx.attr.dep[OutputGroupInfo]",
        "  if hasattr(dep_og, 'a1_group'):",
        "     og['a1_group'] = dep_og.a1_group",
        "  if hasattr(dep_og, 'a2_group'):",
        "     og['a2_group'] = dep_og.a2_group",
        "  return [OutputGroupInfo(**og)]",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })",
        "def _a2_impl(target, ctx):",
        "  g = ctx.actions.declare_file(target.label.name + '_a2.txt')",
        "  ctx.actions.write(g, 'f')",
        "  return [OutputGroupInfo(a2_group = depset([g]))]",
        "",
        "a2 = aspect(implementation=_a2_impl, attr_aspects = ['dep'])",
        "my_rule2 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a2]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1', 'my_rule2')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')",
        "my_rule2(name = 'yyy', dep = ':xxx')");

    AnalysisResult analysisResult = update("//test:yyy");
    OutputGroupInfo outputGroupInfo =
        OutputGroupInfo.get(Iterables.getOnlyElement(analysisResult.getTargetsToBuild()));
    assertThat(getOutputGroupContents(outputGroupInfo, "a1_group"))
        .containsExactly("test/base_a1.txt");
    assertThat(getOutputGroupContents(outputGroupInfo, "a2_group"))
        .containsExactly("test/xxx_a2.txt");
  }

  @Test
  public void duplicateOutputGroupsFromTwoAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target, ctx):",
        "  f = ctx.actions.declare_file(target.label.name + '_a1.txt')",
        "  ctx.actions.write(f, 'f')",
        "  return struct(output_groups = { 'a1_group' : depset([f]) })",
        "",
        "a1 = aspect(implementation=_a1_impl, attr_aspects = ['dep'])",
        "def _rule_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct()",
        "  og = {k:ctx.attr.dep.output_groups[k] for k in ctx.attr.dep.output_groups}",
        "  return struct(output_groups = og)",
        "my_rule1 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a1]) })",
        "def _a2_impl(target, ctx):",
        "  g = ctx.actions.declare_file(target.label.name + '_a2.txt')",
        "  ctx.actions.write(g, 'f')",
        "  return struct(output_groups = { 'a1_group' : depset([g]) })",
        "",
        "a2 = aspect(implementation=_a2_impl, attr_aspects = ['dep'])",
        "my_rule2 = rule(_rule_impl, attrs = { 'dep' : attr.label(aspects = [a2]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule1', 'my_rule2')",
        "my_rule1(name = 'base')",
        "my_rule1(name = 'xxx', dep = ':base')",
        "my_rule2(name = 'yyy', dep = ':xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult analysisResult = update("//test:yyy");
      assertThat(analysisResult.hasError()).isTrue();
      assertThat(keepGoing()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expected.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:9: Output group a1_group provided twice");
  }

  private static Iterable<String> getOutputGroupContents(
      OutputGroupInfo outputGroupInfo, String groupName) {
    return Iterables.transform(
        outputGroupInfo.getOutputGroup(groupName).toList(), Artifact::getRootRelativePathString);
  }

  @Test
  public void duplicateStarlarkProviders() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  return struct(duplicate = 'x')",
        "",
        "MyAspect = aspect(implementation=_impl)",
        "def _rule_impl(ctx):",
        "  return struct(duplicate = 'y')",
        "my_rule = rule(_rule_impl)",
        "def _noop(ctx):",
        "  pass",
        "rbase = rule(_noop, attrs = { 'dep' : attr.label(aspects = [MyAspect]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'my_rule', 'rbase')",
        "my_rule(name = 'xxx')",
        "rbase(name = 'yyy', dep = ':xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update("//test:yyy");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("ERROR /workspace/test/BUILD:3:6: Provider duplicate provided twice");
  }

  @Test
  public void topLevelAspectDoesNotExist() throws Exception {
    scratch.file("test/aspect.bzl", "");
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("MyAspect is not exported from //test:aspect.bzl");
  }

  @Test
  public void topLevelAspectDoesNotExist2() throws Exception {
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("cannot load '//test:aspect.bzl': no such file");
  }

  @Test
  public void topLevelAspectDoesNotExistNoBuildFile() throws Exception {
    scratch.file("test/BUILD", "java_library(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.of("foo/aspect.bzl%MyAspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect to fail.
    }
    assertContainsEvent("Every .bzl file must have a corresponding package");
  }

  @Test
  public void aspectParametersUncovered() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectUncovered = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectUncovered]) },",
        ")");
    scratch.file("test/BUILD", "load('//test:aspect.bzl', 'my_rule')", "my_rule(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent( // "ERROR /workspace/test/aspect.bzl:9:11: "
        "Aspect //test:aspect.bzl%MyAspectUncovered requires rule my_rule to specify attribute "
            + "'my_attr' with type string.");
  }

  @Test
  public void aspectParametersTypeMismatch() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectMismatch = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectMismatch]),",
        "              'my_attr' : attr.int() },",
        ")");
    scratch.file(
        "test/BUILD", "load('//test:aspect.bzl', 'my_rule')", "my_rule(name = 'xxx', my_attr = 4)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(
        "Aspect //test:aspect.bzl%MyAspectMismatch requires rule my_rule to specify attribute "
            + "'my_attr' with type string.");
  }

  @Test
  public void aspectParametersDontSupportSelect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectMismatch = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectMismatch]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = select({'//conditions:default': 'foo'}))");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(
        "//test:xxx: attribute 'my_attr' has a select() and aspect "
            + "//test:aspect.bzl%MyAspectMismatch also declares '//test:xxx'. Aspect attributes "
            + "don't currently support select().");
  }

  @Test
  public void aspectParametersBadDefault() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectBadDefault = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a'], default='b') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectBadDefault]) },",
        ")");
    scratch.file("test/BUILD", "load('//test:aspect.bzl', 'my_rule')", "my_rule(name = 'xxx')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/aspect.bzl:5:28: "
            + "Aspect parameter attribute 'my_attr' has a bad default value: has to be one of 'a' "
            + "instead of 'b'");
  }

  @Test
  public void aspectParametersBadValue() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectBadValue = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectBadValue]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file(
        "test/BUILD", //
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr='b')");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(result.hasError()).isTrue();
    } catch (Exception e) {
      // expect to fail.
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:2:8: //test:xxx: invalid value in 'my_attr' "
            + "attribute: has to be one of 'a' instead of 'b'");
  }

  @Test
  public void aspectParameters() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspect = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa']) },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspect]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = 'aaa')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void aspectParametersOptional() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectOptParam = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['aaa'], default='aaa') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectOptParam]) },",
        ")");
    scratch.file("test/BUILD", "load('//test:aspect.bzl', 'my_rule')", "my_rule(name = 'xxx')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void aspectParametersOptionalOverride() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   if (ctx.attr.my_attr == 'a'):",
        "       fail('Rule is not overriding default, still has value ' + ctx.attr.my_attr)",
        "   return struct()",
        "def _rule_impl(ctx):",
        "   return struct()",
        "MyAspectOptOverride = aspect(",
        "    implementation=_impl,",
        "    attrs = { 'my_attr' : attr.string(values=['a', 'b'], default='a') },",
        ")",
        "my_rule = rule(",
        "    implementation=_rule_impl,",
        "    attrs = { 'deps' : attr.label_list(aspects=[MyAspectOptOverride]),",
        "              'my_attr' : attr.string() },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "my_rule(name = 'xxx', my_attr = 'b')");

    AnalysisResult result = update(ImmutableList.<String>of(), "//test:xxx");
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testMultipleExecutablesInTarget() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct()",
        "my_aspect = aspect(_aspect_impl)",
        "def _main_rule_impl(ctx):",
        "   pass",
        "my_rule = rule(_main_rule_impl,",
        "   attrs = { ",
        "      'exe1' : attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "      'exe2' : attr.label(executable = True, allow_files = True, cfg = 'host'),",
        "   },",
        ")");

    scratch.file("foo/tool.sh", "#!/bin/bash");
    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl',  'my_rule')",
        "my_rule(name = 'main', exe1 = ':tool.sh', exe2 = ':tool.sh')");
    AnalysisResult analysisResultOfRule = update(ImmutableList.<String>of(), "//foo:main");
    assertThat(analysisResultOfRule.hasError()).isFalse();

    AnalysisResult analysisResultOfAspect =
        update(ImmutableList.<String>of("/foo/extension.bzl%my_aspect"), "//foo:main");
    assertThat(analysisResultOfAspect.hasError()).isFalse();
  }

  @Test
  public void aspectFragmentAccessSuccess() throws Exception {
    getConfiguredTargetForAspectFragment(
        "ctx.fragments.java.strict_java_deps", "'java'", "", "", "");
    assertNoEvents();
  }

  @Test
  public void aspectHostFragmentAccessSuccess() throws Exception {
    getConfiguredTargetForAspectFragment(
        "ctx.host_fragments.java.strict_java_deps", "", "'java'", "", "");
    assertNoEvents();
  }

  @Test
  public void aspectFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    assertThrows(
        ViewCreationFailedException.class,
        () ->
            getConfiguredTargetForAspectFragment(
                "ctx.fragments.java.strict_java_deps", "'cpp'", "'java'", "'java'", ""));
    assertContainsEvent(
        "//test:aspect.bzl%MyAspect aspect on my_rule has to declare 'java' as a "
            + "required fragment in target configuration in order to access it. Please update the "
            + "'fragments' argument of the rule definition "
            + "(for example: fragments = [\"java\"])");
  }

  @Test
  public void aspectHostFragmentAccessError() throws Exception {
    reporter.removeHandler(failFastHandler);
    assertThrows(
        ViewCreationFailedException.class,
        () ->
            getConfiguredTargetForAspectFragment(
                "ctx.host_fragments.java.java_strict_deps", "'java'", "'cpp'", "", "'java'"));
    assertContainsEvent(
        "//test:aspect.bzl%MyAspect aspect on my_rule has to declare 'java' as a "
            + "required fragment in host configuration in order to access it. Please update the "
            + "'host_fragments' argument of the rule definition "
            + "(for example: host_fragments = [\"java\"])");
  }

  private ConfiguredTarget getConfiguredTargetForAspectFragment(
      String fullFieldName,
      String fragments,
      String hostFragments,
      String ruleFragments,
      String ruleHostFragments)
      throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "   return struct(result = str(" + fullFieldName + "))",
        "",
        "def _rule_impl(ctx):",
        "   return struct(stuff = '...')",
        "",
        "MyAspect = aspect(",
        "   implementation=_aspect_impl,",
        "   attr_aspects=['deps'],",
        "   fragments=[" + fragments + "],",
        "   host_fragments=[" + hostFragments + "],",
        ")",
        "my_rule = rule(",
        "   implementation=_rule_impl,",
        "   attrs = { 'attr' : ",
        "             attr.label_list(mandatory=True, allow_files=True, aspects = [MyAspect]) },",
        "   fragments=[" + ruleFragments + "],",
        "   host_fragments=[" + ruleHostFragments + "],",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "exports_files(['zzz'])",
        "my_rule(",
        "     name = 'yyy',",
        "     attr = ['zzz'],",
        ")",
        "my_rule(",
        "     name = 'xxx',",
        "     attr = ['yyy'],",
        ")");

    AnalysisResult result = update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    if (result.hasError()) {
      assertThat(keepGoing()).isTrue();
      throw new ViewCreationFailedException("Analysis failed");
    }

    return getConfiguredTarget("//test:xxx");
  }

  @Test
  public void invalidateAspectOnBzlFileChange() throws Exception {
    scratch.file("test/build_defs.bzl", aspectBzlFile("'deps'"));
    scratch.file(
        "test/BUILD",
        "load(':build_defs.bzl', 'repro', 'repro_no_aspect')",
        "repro_no_aspect(name = 'r0')",
        "repro_no_aspect(name = 'r1', deps = [':r0'])",
        "repro(name = 'r2', deps = [':r1'])");
    buildTargetAndCheckRuleInfo("//test:r0", "//test:r1");

    // Make aspect propagation list empty.
    scratch.overwriteFile("test/build_defs.bzl", aspectBzlFile(""));

    // The aspect should not propagate to //test:r0 anymore.
    buildTargetAndCheckRuleInfo("//test:r1");
  }

  private void buildTargetAndCheckRuleInfo(String... expectedLabels) throws Exception {
    AnalysisResult result = update(ImmutableList.<String>of(), "//test:r2");
    ConfiguredTarget configuredTarget = result.getTargetsToBuild().iterator().next();
    Depset ruleInfoValue = (Depset) configuredTarget.get("rule_info");
    assertThat(ruleInfoValue.getSet(String.class).toList())
        .containsExactlyElementsIn(expectedLabels);
  }

  private String[] aspectBzlFile(String attrAspects) {
    return new String[] {
      "def _repro_aspect_impl(target, ctx):",
      "    s = depset([str(target.label)], transitive =",
      "      [d.aspect_info for d in ctx.rule.attr.deps if hasattr(d, 'aspect_info')])",
      "    return struct(aspect_info = s)",
      "",
      "_repro_aspect = aspect(",
      "    _repro_aspect_impl,",
      "    attr_aspects = [" + attrAspects + "],",
      ")",
      "",
      "def repro_impl(ctx):",
      "    s = depset(transitive = ",
      "      [d.aspect_info for d in ctx.attr.deps if hasattr(d, 'aspect_info')])",
      "    return struct(rule_info = s)",
      "",
      "def repro_no_aspect_impl(ctx):",
      "    pass",
      "",
      "repro_no_aspect = rule(implementation = repro_no_aspect_impl,",
      "             attrs = {",
      "                       'deps': attr.label_list(",
      "                             allow_files = True,",
      "                       )",
      "                      },",
      ")",
      "",
      "repro = rule(implementation = repro_impl,",
      "             attrs = {",
      "                       'deps': attr.label_list(",
      "                             allow_files = True,",
      "                             aspects = [_repro_aspect],",
      "                       )",
      "                      },",
      ")"
    };
  }

  @Test
  public void aspectOutputsToBinDirectory() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "def _aspect_impl(target, ctx):",
        "   file = ctx.actions.declare_file('aspect-output-' + target.label.name)",
        "   ctx.actions.write(file, 'data')",
        "   return struct(aspect_file = file)",
        "my_aspect = aspect(_aspect_impl)",
        "def _rule_impl(ctx):",
        "   pass",
        "rule_bin_out = rule(_rule_impl, output_to_genfiles=False)",
        "rule_gen_out = rule(_rule_impl, output_to_genfiles=True)",
        "def _main_rule_impl(ctx):",
        "   s = depset([d.aspect_file for d in ctx.attr.deps])",
        "   return struct(aspect_files = s)",
        "main_rule = rule(_main_rule_impl,",
        "   attrs = { 'deps' : attr.label_list(aspects = [my_aspect]) },",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':extension.bzl', 'rule_bin_out', 'rule_gen_out', 'main_rule')",
        "rule_bin_out(name = 'rbin')",
        "rule_gen_out(name = 'rgen')",
        "main_rule(name = 'main', deps = [':rbin', ':rgen'])");
    AnalysisResult analysisResult = update(ImmutableList.<String>of(), "//foo:main");
    ConfiguredTarget target = analysisResult.getTargetsToBuild().iterator().next();
    NestedSet<Artifact> aspectFiles = ((Depset) target.get("aspect_files")).getSet(Artifact.class);
    assertThat(Iterables.transform(aspectFiles.toList(), Artifact::getFilename))
        .containsExactly("aspect-output-rbin", "aspect-output-rgen");
    for (Artifact aspectFile : aspectFiles.toList()) {
      String rootPath = aspectFile.getRoot().getExecPath().toString();
      assertWithMessage("Artifact %s should not be in genfiles", aspectFile)
          .that(rootPath)
          .doesNotContain("genfiles");
      assertWithMessage("Artifact %s should be in bin", aspectFile).that(rootPath).endsWith("bin");
    }
  }

  @Test
  public void toplevelAspectOnFile() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   print('This aspect does nothing')",
        "   return struct()",
        "MyAspect = aspect(implementation=_impl)");
    scratch.file("test/BUILD", "exports_files(['file.txt'])");
    scratch.file("test/file.txt", "");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:file.txt");
    assertThat(analysisResult.hasError()).isFalse();
    assertThat(
            Iterables.getOnlyElement(analysisResult.getAspectsMap().values())
                .getProviders()
                .getProviderCount())
        .isEqualTo(0);
  }

  @Test
  public void sharedAttributeDefinitionWithAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)",
        "_ATTR = { 'deps' : attr.label_list(aspects = [my_aspect]) }",
        "def _dummy_impl(ctx):",
        "  pass",
        "r1 = rule(_dummy_impl, attrs =  _ATTR)",
        "r2 = rule(_dummy_impl, attrs =  _ATTR)");

    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 't1')",
        "r2(name = 't2', deps = [':t1'])");
    AnalysisResult analysisResult = update("//test:t2");
    assertThat(analysisResult.hasError()).isFalse();
  }

  @Test
  public void multipleAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)",
        "def _dummy_impl(ctx):",
        "  pass",
        "r1 = rule(_dummy_impl, ",
        "          attrs = { 'deps' : attr.label_list(aspects = [my_aspect, my_aspect]) })");

    scratch.file("test/BUILD", "load(':aspect.bzl', 'r1')", "r1(name = 't1')");
    reporter.removeHandler(failFastHandler);
    // This call succeeds if "--keep_going" was passed, which it does in the WithKeepGoing test
    // suite. Otherwise, it fails and throws a TargetParsingException.
    if (keepGoing()) {
      AnalysisResult result = update("//test:r1");
      assertThat(result.hasError()).isTrue();
    } else {
      assertThrows(TargetParsingException.class, () -> update("//test:r1"));
    }
    assertContainsEvent("aspect //test:aspect.bzl%my_aspect added more than once");
  }

  @Test
  public void topLevelAspectsAndExtraActions() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target,ctx):",
        "  f = ctx.actions.declare_file('dummy.txt')",
        "  ctx.actions.run_shell(outputs = [f], command='echo xxx > $(location f)',",
        "                        mnemonic='AspectAction')",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)");
    scratch.file(
        "test/BUILD",
        "extra_action(",
        "    name = 'xa',",
        "    cmd = 'echo $(EXTRA_ACTION_FILE) > $(output file.xa)',",
        "    out_templates = ['file.xa'],",
        ")",
        "action_listener(",
        "    name = 'al',",
        "    mnemonics = [ 'AspectAction' ],",
        "    extra_actions = [ ':xa' ])",
        "java_library(name = 'xxx')");
    useConfiguration("--experimental_action_listener=//test:al");
    AnalysisResult analysisResult =
        update(ImmutableList.<String>of("test/aspect.bzl%my_aspect"), "//test:xxx");
    assertThat(
            Iterables.transform(
                analysisResult.getTopLevelArtifactsToOwnerLabels().getArtifacts(),
                Artifact::getFilename))
        .contains("file.xa");
  }

  /** Regression test for b/137960630. */
  @Test
  public void topLevelAspectsAndExtraActionsWithConflict() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _aspect_impl(target, ctx):",
        "  f = ctx.actions.declare_file('dummy.txt')",
        "  ctx.actions.run_shell(outputs = [f], command='echo xxx > $(location f)',",
        "                        mnemonic='AspectAction')",
        "  return struct()",
        "my_aspect = aspect(implementation = _aspect_impl)");
    scratch.file(
        "test/BUILD",
        "extra_action(",
        "    name = 'xa',",
        "    cmd = 'echo $(EXTRA_ACTION_FILE) > $(output file.xa)',",
        "    out_templates = ['file.xa'],",
        ")",
        "action_listener(",
        "    name = 'al',",
        "    mnemonics = ['AspectAction'],",
        "    extra_actions = [':xa'],",
        ")",
        "java_library(name = 'xxx')",
        "java_library(name = 'yyy')");
    useConfiguration("--experimental_action_listener=//test:al");
    reporter.removeHandler(failFastHandler); // We expect an error.

    if (keepGoing()) {
      AnalysisResult result =
          update(ImmutableList.of("test/aspect.bzl%my_aspect"), "//test:xxx", "//test:yyy");
      assertThat(result.hasError()).isTrue();
    } else {
      assertThrows(
          ViewCreationFailedException.class,
          () -> update(ImmutableList.of("test/aspect.bzl%my_aspect"), "//test:xxx", "//test:yyy"));
    }
    assertContainsEvent(
        "file 'extra_actions/test/xa/test/file.xa' is generated by these conflicting actions");
  }

  @Test
  public void aspectsPropagatingToAllAttributes() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   s = depset([target.label], transitive =",
        "     [i.target_labels for i in ctx.rule.attr.runtime_deps]",
        "     if hasattr(ctx.rule.attr, 'runtime_deps') else [])",
        "   return struct(target_labels = s)",
        "",
        "MyAspect = aspect(",
        "    implementation=_impl,",
        "    attrs = { '_tool' : attr.label(default = Label('//test:tool')) },",
        "    attr_aspects=['*'],",
        ")");
    scratch.file(
        "test/BUILD",
        "java_library(",
        "    name = 'tool',",
        ")",
        "java_library(",
        "     name = 'bar',",
        "     runtime_deps = [':tool'],",
        ")",
        "java_library(",
        "     name = 'foo',",
        "     runtime_deps = [':bar'],",
        ")");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:foo");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    assertThat(configuredAspect).isNotNull();
    Object names = configuredAspect.get("target_labels");
    assertThat(names).isInstanceOf(Depset.class);
    assertThat(
            Iterables.transform(
                ((Depset) names).toList(),
                o -> {
                  assertThat(o).isInstanceOf(Label.class);
                  return ((Label) o).getName();
                }))
        .containsExactly("foo", "bar", "tool");
  }

  /** Simple straightforward linear aspects-on-aspects. */
  @Test
  public void aspectOnAspectLinear() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  if hasattr(target, 'a1p'):",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=yes')",
        "  else:",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=no')",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence) target.get("result");

    // "yes" means that aspect a2 sees a1's providers.
    assertThat(result)
        .containsExactly(
            "//test:r0[\"//test:aspect.bzl%a1\", \"//test:aspect.bzl%a2\"]=yes",
            "//test:r1[\"//test:aspect.bzl%a2\"]=no");
  }

  /**
   * Diamond case. rule r1 depends or r0 with aspect a1. rule r2 depends or r0 with aspect a2. rule
   * rcollect depends on r1, r2 with aspect a3.
   *
   * <p>Aspect a3 should be applied twice to target r0: once in [a1, a3] sequence and once in [a2,
   * a3] sequence.
   */
  @Test
  public void aspectOnAspectDiamond() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = 'text from a1')",
        "a1 = aspect(_a1_impl, attr_aspects = ['deps'], provides = ['a1p'])",
        "",
        "def _a2_impl(target,ctx):",
        "  return struct(a2p = 'text from a2')",
        "a2 = aspect(_a2_impl, attr_aspects = ['deps'], provides = ['a2p'])",
        "",
        "def _a3_impl(target,ctx):",
        "  value = []",
        "  f = ctx.actions.declare_file('a3.out')",
        "  ctx.actions.write(f, 'text')",
        "  for dep in ctx.rule.attr.deps:",
        "     if hasattr(dep, 'a3p'):",
        "         value += dep.a3p",
        "  s = str(target.label) + str(ctx.aspect_ids) + '='",
        "  if hasattr(target, 'a1p'):",
        "     s += 'a1p'",
        "  if hasattr(target, 'a2p'):",
        "     s += 'a2p'",
        "  value.append(s)",
        "  return struct(a3p = value)",
        "a3 = aspect(_a3_impl, attr_aspects = ['deps'],",
        "            required_aspect_providers = [['a1p'], ['a2p']])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _rcollect_impl(ctx):",
        "  value = []",
        "  for dep in ctx.attr.deps:",
        "     if hasattr(dep, 'a3p'):",
        "         value += dep.a3p",
        "  return struct(result = value)",
        "r1 = rule(_r1_impl, attrs = { 'deps' : attr.label_list(aspects = [a1])})",
        "r2 = rule(_r1_impl, attrs = { 'deps' : attr.label_list(aspects = [a2])})",
        "rcollect = rule(_rcollect_impl, attrs = { 'deps' : attr.label_list(aspects = [a3])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2', 'rcollect')",
        "r1(name = 'r0')",
        "r1(name = 'r1', deps = [':r0'])",
        "r2(name = 'r2', deps = [':r0'])",
        "rcollect(name = 'rcollect', deps = [':r1', ':r2'])");
    AnalysisResult analysisResult = update("//test:rcollect");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence) target.get("result");
    assertThat(result)
        .containsExactly(
            "//test:r0[\"//test:aspect.bzl%a1\", \"//test:aspect.bzl%a3\"]=a1p",
            "//test:r1[\"//test:aspect.bzl%a3\"]=",
            "//test:r0[\"//test:aspect.bzl%a2\", \"//test:aspect.bzl%a3\"]=a2p",
            "//test:r2[\"//test:aspect.bzl%a3\"]=");
  }

  /**
   * Linear with duplicates. r2_1 depends on r0 with aspect a2. r1 depends on r2_1 with aspect a1.
   * r2 depends on r1 with aspect a2.
   *
   * <p>a2 is not interested in a1. There should be just one instance of aspect a2 on r0, and is
   * should *not* see a1.
   */
  @Test
  public void aspectOnAspectLinearDuplicates() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = 'a1p')",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  if hasattr(target, 'a1p'):",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=yes')",
        "  else:",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=no')",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = [])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r2(name = 'r2_1', dep = ':r0')",
        "r1(name = 'r1', dep = ':r2_1')",
        "r2(name = 'r2', dep = ':r1')");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence) target.get("result");
    // "yes" means that aspect a2 sees a1's providers.
    assertThat(result)
        .containsExactly(
            "//test:r0[\"//test:aspect.bzl%a2\"]=no",
            "//test:r1[\"//test:aspect.bzl%a2\"]=no", "//test:r2_1[\"//test:aspect.bzl%a2\"]=no");
  }

  /** Linear aspects-on-aspects with alias rule. */
  @Test
  public void aspectOnAspectLinearAlias() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  if hasattr(target, 'a1p'):",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=yes')",
        "  else:",
        "     value.append(str(target.label) + str(ctx.aspect_ids) + '=no')",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "alias(name = 'a0', actual = ':r0')",
        "r1(name = 'r1', dep = ':a0')",
        "r2(name = 'r2', dep = ':r1')");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence<?>) target.get("result");

    // "yes" means that aspect a2 sees a1's providers.
    assertThat(result)
        .containsExactly(
            "//test:r0[\"//test:aspect.bzl%a1\", \"//test:aspect.bzl%a2\"]=yes",
            "//test:r1[\"//test:aspect.bzl%a2\"]=no");
  }

  @Test
  public void aspectDescriptions() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _a_impl(target,ctx):",
        "  s = str(target.label) + str(ctx.aspect_ids) + '='",
        "  value = []",
        "  if ctx.rule.attr.dep:",
        "     d = ctx.rule.attr.dep",
        "     this_id = ctx.aspect_ids[len(ctx.aspect_ids) - 1]",
        "     s += str(d.label) + str(d.my_ids) + ',' + str(this_id in d.my_ids)",
        "     value += ctx.rule.attr.dep.ap",
        "  else:",
        "     s += 'None'",
        "  value.append(s)",
        "  return struct(ap = value, my_ids = ctx.aspect_ids)",
        "a = aspect(_a_impl, attr_aspects = ['dep'])",
        "def _r_impl(ctx):",
        "  if not ctx.attr.dep:",
        "     return struct(result = [])",
        "  return struct(result = ctx.attr.dep.ap)",
        "r = rule(_r_impl, attrs = { 'dep' : attr.label(aspects = [a])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r')",
        "r(name = 'r0')",
        "r(name = 'r1', dep = ':r0')",
        "r(name = 'r2', dep = ':r1')");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence<?>) target.get("result");

    assertThat(result)
        .containsExactly(
            "//test:r0[\"//test:aspect.bzl%a\"]=None",
            "//test:r1[\"//test:aspect.bzl%a\"]=//test:r0[\"//test:aspect.bzl%a\"],True");
  }

  @Test
  public void attributesWithAspectsReused() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "my_aspect = aspect(_impl)",
        "a_dict = { 'foo' : attr.label_list(aspects = [my_aspect]) }");

    scratch.file(
        "test/r1.bzl",
        "load(':aspect.bzl', 'my_aspect', 'a_dict')",
        "def _rule_impl(ctx):",
        "   pass",
        "r1 = rule(_rule_impl, attrs = a_dict)");

    scratch.file(
        "test/r2.bzl",
        "load(':aspect.bzl', 'my_aspect', 'a_dict')",
        "def _rule_impl(ctx):",
        "   pass",
        "r2 = rule(_rule_impl, attrs = a_dict)");

    scratch.file(
        "test/BUILD",
        "load(':r1.bzl', 'r1')",
        "load(':r2.bzl', 'r2')",
        "r1(name = 'x1')",
        "r2(name = 'x2', foo = [':x1'])");
    AnalysisResult analysisResult = update("//test:x2");
    assertThat(analysisResult.hasError()).isFalse();
  }

  @Test
  public void aspectAdvertisingProviders() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "   return struct()",
        "my_aspect = aspect(_impl, provides = ['foo'])",
        "a_dict = { 'foo' : attr.label_list(aspects = [my_aspect]) }");
    scratch.file("test/BUILD", "java_library(name = 'xxx',)");

    reporter.removeHandler(failFastHandler);
    try {
      AnalysisResult analysisResult =
          update(ImmutableList.of("//test:aspect.bzl%my_aspect"), "//test:xxx");
      assertThat(keepGoing()).isTrue();
      assertThat(analysisResult.hasError()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expect exception
    }
    assertContainsEvent(
        "Aspect '//test:aspect.bzl%my_aspect', applied to '//test:xxx', "
            + "does not provide advertised provider 'foo'");
  }

  @Test
  public void aspectOnAspectInconsistentVisibility() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  return struct(a2p = a2p(value = 'random'))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')",
        "r1(name = 'r1_1', dep = ':r2')",
        "r2(name = 'r2_1', dep = ':r1_1')");
    reporter.removeHandler(failFastHandler);

    try {
      AnalysisResult analysisResult = update("//test:r2_1");
      assertThat(analysisResult.hasError()).isTrue();
      assertThat(keepGoing()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expected
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:3:3: Aspect //test:aspect.bzl%a2 is"
            + " applied twice, both before and after aspect //test:aspect.bzl%a1 "
            + "(when propagating to //test:r1)");
  }

  @Test
  public void aspectOnAspectInconsistentVisibilityIndirect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "  return struct(a2p = a2p(value = 'random'))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})",
        "def _r0_impl(ctx):",
        "  pass",
        "r0 = rule(_r0_impl, attrs = { 'dep' : attr.label()})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r1', 'r2')",
        "r0(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')",
        "r1(name = 'r1_1', dep = ':r2')",
        "r2(name = 'r2_1', dep = ':r1_1')",
        "r0(name = 'r0_2', dep = ':r2_1')");
    reporter.removeHandler(failFastHandler);

    try {
      AnalysisResult analysisResult = update("//test:r0_2");
      assertThat(analysisResult.hasError()).isTrue();
      assertThat(keepGoing()).isTrue();
    } catch (ViewCreationFailedException e) {
      // expected
    }
    assertContainsEvent(
        "ERROR /workspace/test/BUILD:3:3: Aspect //test:aspect.bzl%a2 is"
            + " applied twice, both before and after aspect //test:aspect.bzl%a1 "
            + "(when propagating to //test:r1)");
  }

  /**
   * Aspect a3 sees aspect a2, aspect a2 sees aspect a1, but a3 does not see a1. All three aspects
   * should still propagate together.
   */
  @Test
  public void aspectOnAspectOnAspect() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "p1 = provider()",
        "def _a1_impl(target, ctx):",
        "   return [p1()]",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = [p1])",
        "p2 = provider()",
        "def _a2_impl(target, ctx):",
        "   value = True if p1 in target else False",
        "   return [p2(has_p1 = value)]",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'],",
        "   required_aspect_providers = [p1], provides = [p2])",
        "p3 = provider()",
        "def _a3_impl(target, ctx):",
        "   list = []",
        "   if ctx.rule.attr.dep:",
        "     list = ctx.rule.attr.dep[p3].value",
        "   my_value = str(target.label) +'=' + str(target[p2].has_p1 if p2 in target else False)",
        "   return [p3(value = list + [my_value])]",
        "a3 = aspect(_a3_impl, attr_aspects = ['dep'],",
        "   required_aspect_providers = [p2])",
        "def _r0_impl(ctx):",
        "  pass",
        "r0 = rule(_r0_impl, attrs = { 'dep' : attr.label()})",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  pass",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r1', 'r2')",
        "r0(name = 'r0_1')",
        "r0(name = 'r0_2', dep = ':r0_1')",
        "r0(name = 'r0_3', dep = ':r0_2')",
        "r1(name = 'r1_1', dep = ':r0_3')",
        "r2(name = 'r2_1', dep = ':r1_1')");

    AnalysisResult analysisResult = update(ImmutableList.of("//test:aspect.bzl%a3"), "//test:r2_1");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key p3 =
        new StarlarkProvider.Key(Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "p3");
    StructImpl p3Provider = (StructImpl) configuredAspect.get(p3);
    assertThat((Sequence<?>) p3Provider.getValue("value"))
        .containsExactly(
            "//test:r0_1=True",
            "//test:r0_2=True",
            "//test:r0_3=True",
            "//test:r1_1=False",
            "//test:r2_1=False");
  }

  /**
   * r0 is a dependency of r1 via two attributes, dep1 and dep2. r1 sends an aspect 'a' along dep1
   * but not along dep2.
   *
   * <p>rcollect depends upon r1 and sends another aspect, 'collector', along its dep dependency.
   * 'collector' wants to see aspect 'a' and propagates along dep1 and dep2. It should be applied
   * both to r0 and to r0+a.
   */
  @Test
  public void multipleDepsDifferentAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "PAspect = provider()",
        "PCollector = provider()",
        "def _aspect_impl(target, ctx):",
        "   return [PAspect()]",
        "a = aspect(_aspect_impl, attr_aspects = ['dep'], provides = [PAspect])",
        "def _collector_impl(target, ctx):",
        "   suffix = '+PAspect' if PAspect in target else ''",
        "   result = [str(target.label)+suffix]",
        "   for a in ['dep', 'dep1', 'dep2']:",
        "     if hasattr(ctx.rule.attr, a):",
        "        result += getattr(ctx.rule.attr, a)[PCollector].result",
        "   return [PCollector(result=result)]",
        "collector = aspect(_collector_impl, attr_aspects = ['*'], ",
        "                   required_aspect_providers = [PAspect])",
        "def _rimpl(ctx):",
        "   pass",
        "r0 = rule(_rimpl)",
        "r1 = rule(_rimpl, ",
        "          attrs = {",
        "             'dep1' : attr.label(),",
        "             'dep2' : attr.label(aspects = [a]),",
        "          },",
        ")",
        "def _rcollect_impl(ctx):",
        "    return [ctx.attr.dep[PCollector]]",
        "rcollect = rule(_rcollect_impl,",
        "                attrs = {",
        "                  'dep' : attr.label(aspects = [collector]),",
        "                })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r1', 'rcollect')",
        "r0(name = 'r0')",
        "r1(name = 'r1', dep1 = ':r0', dep2 = ':r0')",
        "rcollect(name = 'rcollect', dep = ':r1')");

    AnalysisResult analysisResult = update(ImmutableList.of(), "//test:rcollect");
    ConfiguredTarget configuredTarget =
        Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    StarlarkProvider.Key pCollector =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "PCollector");
    StructImpl pCollectorProvider = (StructImpl) configuredTarget.get(pCollector);
    assertThat((Sequence<?>) pCollectorProvider.getValue("result"))
        .containsExactly("//test:r1", "//test:r0", "//test:r0+PAspect");
  }

  @Test
  public void aspectSeesOtherAspectAttributes() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "PAspect = provider(fields = [])",
        "PCollector = provider(fields = ['aspect_attr'])",
        "def _a_impl(target, ctx):",
        "  return [PAspect()]",
        "a = aspect(_a_impl, ",
        "           provides = [PAspect],",
        "           attrs = {'_a_attr' : attr.label(default = '//test:foo')})",
        "def _rcollect(target, ctx):",
        "  if hasattr(ctx.rule.attr, '_a_attr'):",
        "     return [PCollector(aspect_attr = ctx.rule.attr._a_attr.label)]",
        "  if hasattr(ctx.rule.attr, 'dep'):",
        "     return [ctx.rule.attr.dep[PCollector]]",
        "  return [PCollector()]",
        "acollect = aspect(_rcollect, attr_aspects = ['*'], required_aspect_providers = [PAspect])",
        "def _rimpl(ctx):",
        "  pass",
        "r0 = rule(_rimpl)",
        "r = rule(_rimpl, attrs = { 'dep' : attr.label(aspects = [a]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r')",
        "r0(name = 'foo')",
        "r0(name = 'bar')",
        "r(name = 'baz', dep = ':bar')");
    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%acollect"), "//test:baz");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key pCollector =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "PCollector");
    StructImpl collector = (StructImpl) configuredAspect.get(pCollector);
    assertThat(collector.getValue("aspect_attr"))
        .isEqualTo(Label.parseAbsolute("//test:foo", ImmutableMap.of()));
  }

  @Test
  public void ruleAttributesWinOverAspects() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "PAspect = provider(fields = [])",
        "PCollector = provider(fields = ['attr_value'])",
        "def _a_impl(target, ctx):",
        "  return [PAspect()]",
        "a = aspect(_a_impl, ",
        "           provides = [PAspect],",
        "           attrs = {'_same_attr' : attr.int(default = 239)})",
        "def _rcollect(target, ctx):",
        "  if hasattr(ctx.rule.attr, '_same_attr'):",
        "     return [PCollector(attr_value = ctx.rule.attr._same_attr)]",
        "  if hasattr(ctx.rule.attr, 'dep'):",
        "     return [ctx.rule.attr.dep[PCollector]]",
        "  return [PCollector()]",
        "acollect = aspect(_rcollect, attr_aspects = ['*'], required_aspect_providers = [PAspect])",
        "def _rimpl(ctx):",
        "  pass",
        "r0 = rule(_rimpl)",
        "r = rule(_rimpl, ",
        "          attrs = { ",
        "                  'dep' : attr.label(aspects = [a]), ",
        "                  '_same_attr' : attr.int(default = 30)",
        "          })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r')",
        "r0(name = 'foo')",
        "r0(name = 'bar')",
        "r(name = 'baz', dep = ':bar')");
    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%acollect"), "//test:baz");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key pCollector =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "PCollector");
    StructImpl collector = (StructImpl) configuredAspect.get(pCollector);
    assertThat(collector.getValue("attr_value")).isEqualTo(30);
  }

  @Test
  public void earlyAspectAttributesWin() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "PAspect1 = provider(fields = [])",
        "PAspect2 = provider(fields = [])",
        "PCollector = provider(fields = ['attr_value'])",
        "def _a1_impl(target, ctx):",
        "  return [PAspect1()]",
        "def _a2_impl(target, ctx):",
        "  return [PAspect2()]",
        "a1 = aspect(_a1_impl, ",
        "            provides = [PAspect1],",
        "            attrs = {'_same_attr' : attr.int(default = 30)})",
        "a2 = aspect(_a2_impl, ",
        "            provides = [PAspect2],",
        "            attrs = {'_same_attr' : attr.int(default = 239)})",
        "def _rcollect(target, ctx):",
        "  if hasattr(ctx.rule.attr, 'dep'):",
        "     return [ctx.rule.attr.dep[PCollector]]",
        "  if hasattr(ctx.rule.attr, '_same_attr'):",
        "     return [PCollector(attr_value = ctx.rule.attr._same_attr)]",
        "  fail('???')",
        "  return [PCollector()]",
        "acollect = aspect(_rcollect, attr_aspects = ['*'], ",
        "                  required_aspect_providers = [[PAspect1], [PAspect2]])",
        "def _rimpl(ctx):",
        "  pass",
        "r0 = rule(_rimpl)",
        "r1 = rule(_rimpl, ",
        "          attrs = { ",
        "                  'dep' : attr.label(aspects = [a1]), ",
        "          })",
        "r2 = rule(_rimpl, ",
        "          attrs = { ",
        "                  'dep' : attr.label(aspects = [a2]), ",
        "          })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r1', 'r2')",
        "r0(name = 'bar')",
        "r1(name = 'baz', dep = ':bar')",
        "r2(name = 'quux', dep = ':baz')");

    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%acollect"), "//test:quux");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key pCollector =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "PCollector");
    StructImpl collector = (StructImpl) configuredAspect.get(pCollector);
    assertThat(collector.getValue("attr_value")).isEqualTo(30);
  }

  @Test
  public void aspectPropagatesOverOtherAspectAttributes() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "PAspect = provider(fields = [])",
        "PCollector = provider(fields = ['visited'])",
        "def _a_impl(target, ctx):",
        "  return [PAspect()]",
        "a = aspect(_a_impl, ",
        "       provides = [PAspect],",
        "       attrs = {'_a_attr' : attr.label(default = '//test:referenced_from_aspect_only')})",
        "def _rcollect(target, ctx):",
        "  transitive = []",
        "  if hasattr(ctx.rule.attr, 'dep') and ctx.rule.attr.dep:",
        "     transitive += [ctx.rule.attr.dep[PCollector].visited]",
        "  if hasattr(ctx.rule.attr, '_a_attr') and ctx.rule.attr._a_attr:",
        "     transitive += [ctx.rule.attr._a_attr[PCollector].visited] ",
        "  visited = depset([target.label], transitive = transitive, )",
        "  return [PCollector(visited = visited)]",
        "acollect = aspect(_rcollect, attr_aspects = ['*'], required_aspect_providers = [PAspect])",
        "def _rimpl(ctx):",
        "  pass",
        "r0 = rule(_rimpl)",
        "r = rule(_rimpl, attrs = { 'dep' : attr.label(aspects = [a]) })");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r')",
        "r0(name = 'referenced_from_aspect_only')",
        "r0(name = 'bar')",
        "r(name = 'baz', dep = ':bar')");
    AnalysisResult analysisResult =
        update(ImmutableList.of("//test:aspect.bzl%acollect"), "//test:baz");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key pCollector =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "PCollector");
    StructImpl collector = (StructImpl) configuredAspect.get(pCollector);
    assertThat(((Depset) collector.getValue("visited")).toList())
        .containsExactly(
            Label.parseAbsolute("//test:referenced_from_aspect_only", ImmutableMap.of()),
            Label.parseAbsolute("//test:bar", ImmutableMap.of()),
            Label.parseAbsolute("//test:baz", ImmutableMap.of()));
  }

  @Test
  // This test verifies that aspects which are defined natively and exported for use in Starlark
  // can be referenced at the top level using the --aspects flag. For ease of testing,
  // apple_common.objc_proto_aspect is used as an example.
  public void testTopLevelStarlarkObjcProtoAspect() throws Exception {
    MockObjcSupport.setupObjcProtoLibrary(scratch);
    scratch.file("test_skylark/BUILD");
    scratch.file("x/data_filter.pbascii");
    scratch.file(
        "test_skylark/top_level_stub.bzl",
        "top_level_aspect = apple_common.objc_proto_aspect",
        "",
        "def top_level_stub_impl(ctx):",
        "  return struct()",
        "top_level_stub = rule(",
        "    top_level_stub_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(),",
        "    },",
        "    fragments = ['apple'],",
        ")");

    scratch.file(
        "x/BUILD",
        "load('//objc_proto_library:objc_proto_library.bzl', 'objc_proto_library')",
        "proto_library(",
        "  name = 'protos',",
        "  srcs = ['data.proto'],",
        MockProtoSupport.MIGRATION_TAG,
        ")",
        "objc_proto_library(",
        "  name = 'x',",
        "  deps = [':protos'],",
        "  portable_proto_filters = ['data_filter.pbascii'],",
        ")");

    scratch.file(
        "bin/BUILD",
        "load('//test_skylark:top_level_stub.bzl', 'top_level_stub')",
        "top_level_stub(",
        "  name = 'link_target',",
        "  deps = ['//x:x'],",
        ")");

    useConfiguration(MockObjcSupport.requiredObjcCrosstoolFlags().toArray(new String[1]));
    AnalysisResult analysisResult =
        update(
            ImmutableList.of("test_skylark/top_level_stub.bzl%top_level_aspect"),
            "//bin:link_target");
    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());

    ObjcProtoProvider objcProtoProvider =
        (ObjcProtoProvider) configuredAspect.get(ObjcProtoProvider.STARLARK_CONSTRUCTOR.getKey());
    assertThat(objcProtoProvider).isNotNull();
  }

  @Test
  public void testAspectActionProvider() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  ctx.actions.run_shell(",
        "    outputs = [ctx.actions.declare_file('a1')],",
        "    command = 'touch $@'",
        "  )",
        "  return struct(a1p=a1p())",
        "a1 = aspect(_a1_impl, attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target, ctx):",
        "  value = []",
        "  if hasattr(ctx.rule.attr, 'dep') and hasattr(ctx.rule.attr.dep, 'a2p'):",
        "     value += ctx.rule.attr.dep.a2p.value",
        "  value += target.actions",
        "  return struct(a2p = a2p(value = value))",
        "a2 = aspect(_a2_impl, attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r0_impl(ctx):",
        "  ctx.actions.run_shell(",
        "    outputs = [ctx.actions.declare_file('r0')],",
        "    command = 'touch $@'",
        "  )",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep.a2p.value)",
        "r0 = rule(_r0_impl)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r0', 'r1', 'r2')",
        "r0(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    Sequence<?> result = (Sequence<?>) target.get("result");

    // We should see both the action from the 'r0' rule, and the action from the 'a1' aspect
    assertThat(result).hasSize(2);
    assertThat(
            result.stream()
                .map(a -> ((Action) a).getPrimaryOutput().getExecPath().getBaseName())
                .collect(toList()))
        .containsExactly("r0", "a1");
  }

  @Test
  public void testRuleAndAspectAttrConflict() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "MyInfo = provider()",
        "def _impl(target, ctx):",
        "   return [MyInfo(hidden_attr_label = str(ctx.attr._hiddenattr.label))]",
        "",
        "def _rule_impl(ctx):",
        "   return []",
        "",
        "my_rule = rule(implementation = _rule_impl,",
        "   attrs = { '_hiddenattr' : attr.label(default = Label('//test:xxx')) },",
        ")",
        "MyAspect = aspect(",
        "   implementation=_impl,",
        "   attrs = { '_hiddenattr' : attr.label(default = Label('//test:zzz')) },",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:aspect.bzl', 'my_rule')",
        "cc_library(",
        "     name = 'xxx',",
        ")",
        "my_rule(",
        "     name = 'yyy',",
        ")",
        "cc_library(",
        "     name = 'zzz',",
        ")");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:yyy");
    ConfiguredAspect configuredAspect = analysisResult.getAspectsMap().values().iterator().next();
    assertThat(configuredAspect).isNotNull();

    StarlarkProvider.Key myInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:aspect.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl myInfo = (StructImpl) configuredAspect.get(myInfoKey);
    assertThat(myInfo.getValue("hidden_attr_label")).isEqualTo("//test:zzz");
  }

  /** Simple straightforward linear aspects-on-aspects. */
  @Test
  public void aspectOnAspectAttrConflict() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "MyInfo = provider()",
        "a1p = provider()",
        "def _a1_impl(target,ctx):",
        "  return struct(a1p = a1p(text = 'random'))",
        "a1 = aspect(_a1_impl,",
        "   attrs = { '_hiddenattr' : attr.label(default = Label('//test:xxx')) },",
        "   attr_aspects = ['dep'], provides = ['a1p'])",
        "a2p = provider()",
        "def _a2_impl(target,ctx):",
        "   return [MyInfo(hidden_attr_label = str(ctx.attr._hiddenattr.label))]",
        "a2 = aspect(_a2_impl,",
        "  attrs = { '_hiddenattr' : attr.label(default = Label('//test:zzz')) },",
        "  attr_aspects = ['dep'], required_aspect_providers = ['a1p'])",
        "def _r1_impl(ctx):",
        "  pass",
        "def _r2_impl(ctx):",
        "  return struct(result = ctx.attr.dep[MyInfo].hidden_attr_label)",
        "r1 = rule(_r1_impl, attrs = { 'dep' : attr.label(aspects = [a1])})",
        "r2 = rule(_r2_impl, attrs = { 'dep' : attr.label(aspects = [a2])})");
    scratch.file(
        "test/BUILD",
        "load(':aspect.bzl', 'r1', 'r2')",
        "r1(name = 'r0')",
        "r1(name = 'r1', dep = ':r0')",
        "r2(name = 'r2', dep = ':r1')",
        "cc_library(",
        "     name = 'xxx',",
        ")",
        "cc_library(",
        "     name = 'zzz',",
        ")");
    AnalysisResult analysisResult = update("//test:r2");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    String result = (String) target.get("result");

    assertThat(result).isEqualTo("//test:zzz");
  }

  @Test
  public void testAllCcLibraryAttrsAreValidTypes() throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "def _impl(target, ctx):",
        "  for entry in dir(ctx.rule.attr):",
        "    val = getattr(ctx.rule.attr, entry, None)",
        "    # Only legitimate Starlark values can be passed to dir(), so this effectively",
        "    # verifies val is an appropriate Starlark type.",
        "    _test_dir = dir(val)",
        "  return []",
        "",
        "MyAspect = aspect(",
        "  implementation=_impl,",
        ")");
    scratch.file("test/BUILD", "cc_library(", "     name = 'xxx',", ")");
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");
    assertThat(analysisResult.getAspectsMap().values().iterator().next()).isNotNull();
  }

  @Test
  public void testApplyToGeneratingRules() throws Exception {
    // Create test rules:
    // dep_rule: a rule which may depend on other dep_rule targets and may optionally create
    //     an output file.
    // root_{with,no}_files: a rule which depends on dep_rule targets and attaches an aspect.
    //     The rule returns a RootInfo provider which contains two fields:
    //        'from_aspect' : a list of all labels that the aspect propagated to
    //        'non_aspect' : a list of all labels that information was obtained from without aspect
    //     root_with_files uses an aspect with apply_to_generating_rules=True, and root_no_files
    //     uses an aspect with apply_to_generating_rules=False.
    scratch.file(
        "test/lib.bzl",
        "RootInfo = provider()",
        "NonAspectInfo = provider()",
        "FromAspectInfo = provider()",
        "def _aspect_impl(target, ctx):",
        "  dep_labels = []",
        "  for dep in ctx.rule.attr.deps:",
        "    if FromAspectInfo in dep:",
        "      dep_labels += [dep[FromAspectInfo].labels]",
        "  return FromAspectInfo(labels = depset(direct = [ctx.label], transitive = dep_labels))",
        "",
        "def _rule_impl(ctx):",
        "  non_aspect = []",
        "  from_aspect = []",
        "  for dep in ctx.attr.deps:",
        "    if NonAspectInfo in dep:",
        "      non_aspect +=  dep[NonAspectInfo].labels.to_list()",
        "    if FromAspectInfo in dep:",
        "      from_aspect += dep[FromAspectInfo].labels.to_list()",
        "  return RootInfo(from_aspect = from_aspect, non_aspect = non_aspect)",
        "",
        "def _dep_rule_impl(ctx):",
        "  if ctx.outputs.output:",
        "    ctx.actions.run_shell(outputs = [ctx.outputs.output], command = 'dont run me')",
        "  dep_labels = []",
        "  for dep in ctx.attr.deps:",
        "    if NonAspectInfo in dep:",
        "      dep_labels += [dep[NonAspectInfo].labels]",
        "  return NonAspectInfo(labels = depset(direct = [ctx.label], transitive = dep_labels))",
        "",
        "aspect_with_files = aspect(",
        "  implementation = _aspect_impl,",
        "  attr_aspects = ['deps'],",
        "  apply_to_generating_rules = True)",
        "",
        "aspect_no_files = aspect(",
        "  implementation = _aspect_impl,",
        "  attr_aspects = ['deps'],",
        "  apply_to_generating_rules = False)",
        "",
        "root_with_files = rule(implementation = _rule_impl,",
        "  attrs = {'deps' : attr.label_list(aspects = [aspect_with_files])})",
        "",
        "root_no_files = rule(implementation = _rule_impl,",
        "  attrs = {'deps' : attr.label_list(aspects = [aspect_no_files])})",
        "",
        "dep_rule = rule(implementation = _dep_rule_impl,",
        "  attrs = {'deps' : attr.label_list(allow_files = True), 'output' : attr.output()})");

    // Create a target graph such that two graph roots each point to a common subgraph
    // alpha -> beta_output -> charlie, where beta_output is a generated output file of target
    // 'beta'.
    scratch.file(
        "test/BUILD",
        "load('//test:lib.bzl', 'root_with_files', 'root_no_files', 'dep_rule')",
        "",
        "root_with_files(name = 'test_with_files', deps = [':alpha'])",
        "root_no_files(name = 'test_no_files', deps = [':alpha'])",
        "dep_rule(name = 'alpha', deps = [':beta_output'])",
        "dep_rule(name = 'beta', deps = [':charlie'], output = 'beta_output')",
        "dep_rule(name = 'charlie')");

    StarlarkProvider.Key rootInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "RootInfo");

    AnalysisResult analysisResultWithFiles = update("//test:test_with_files");
    ConfiguredTarget targetWithFiles =
        Iterables.getOnlyElement(analysisResultWithFiles.getTargetsToBuild());
    StructImpl rootInfoWithFiles = (StructImpl) targetWithFiles.get(rootInfoKey);
    // With apply_to_generating_rules=True, the aspect should have traversed :beta_output and
    // applied to both :beta and :charlie.
    assertThat(rootInfoWithFiles.getValue("from_aspect", Sequence.class))
        .containsExactly(
            Label.parseAbsolute("//test:charlie", ImmutableMap.of()),
            Label.parseAbsolute("//test:beta", ImmutableMap.of()),
            Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
    assertThat(rootInfoWithFiles.getValue("non_aspect", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));

    AnalysisResult analysisResultNoFiles = update("//test:test_no_files");
    ConfiguredTarget targetNoFiles =
        Iterables.getOnlyElement(analysisResultNoFiles.getTargetsToBuild());
    StructImpl rootInfoNoFiles = (StructImpl) targetNoFiles.get(rootInfoKey);
    // With apply_to_generating_rules=False, the aspect should have only accessed :alpha, as it
    // must have stopped before :beta_output.
    assertThat(rootInfoNoFiles.getValue("from_aspect", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
    assertThat(rootInfoWithFiles.getValue("non_aspect", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
  }

  private void setupAspectOnAspectTargetGraph(
      boolean applyRootToGeneratingRules, boolean applyDepToGeneratingRules) throws Exception {
    // RootAspectInfo.both_labels returns a list of target labels which
    //     were evaluated as [root_aspect(dep_aspect(target))].
    // RootAspectInfo.root_only_labels returns a list of target labels which
    //     were evaluated as [root_aspect(target)].
    // DepAspectInfo.labels returns a list of target labels which were evaluated by dep_aspect.
    scratch.file(
        "test/lib.bzl",
        "RootAspectInfo = provider()",
        "DepAspectInfo = provider()",
        "def _root_aspect_impl(target, ctx):",
        "  both_labels = []",
        "  root_only_labels = []",
        "  for dep in ctx.rule.attr.deps:",
        "    if RootAspectInfo in dep:",
        "      both_labels += dep[RootAspectInfo].both_labels",
        "      root_only_labels += dep[RootAspectInfo].root_only_labels",
        "      if DepAspectInfo in dep:",
        "        both_labels += [dep.label]",
        "      else:",
        "        root_only_labels += [dep.label]",
        "  return RootAspectInfo(both_labels = both_labels, root_only_labels = root_only_labels)",
        "",
        "def _dep_aspect_impl(target, ctx):",
        "  dep_labels = [ctx.label]",
        "  for dep in ctx.rule.attr.deps:",
        "    if DepAspectInfo in dep:",
        "      dep_labels += dep[DepAspectInfo].labels",
        "  return DepAspectInfo(labels = dep_labels)",
        "",
        "def _root_rule_impl(ctx):",
        "  return [ctx.attr.deps[0][RootAspectInfo], ctx.attr.deps[0][DepAspectInfo]]",
        "",
        "def _dep_with_aspect_rule_impl(ctx):",
        "  return [ctx.attr.deps[0][DepAspectInfo]]",
        "",
        "def _dep_rule_impl(ctx):",
        "  if ctx.outputs.output:",
        "    ctx.actions.run_shell(outputs = [ctx.outputs.output], command = 'dont run me')",
        "  return []",
        "",
        "root_aspect = aspect(",
        "  implementation = _root_aspect_impl,",
        "  attr_aspects = ['deps'],",
        "  required_aspect_providers = [DepAspectInfo],",
        "  apply_to_generating_rules = " + (applyRootToGeneratingRules ? "True" : "False") + ")",
        "",
        "dep_aspect = aspect(",
        "  implementation = _dep_aspect_impl,",
        "  attr_aspects = ['deps'],",
        "  provides = [DepAspectInfo],",
        "  apply_to_generating_rules = " + (applyDepToGeneratingRules ? "True" : "False") + ")",
        "",
        "root_rule = rule(implementation = _root_rule_impl,",
        "  attrs = {'deps' : attr.label_list(aspects = [root_aspect])})",
        "",
        "dep_with_aspect_rule = rule(implementation = _dep_with_aspect_rule_impl,",
        "  attrs = {'deps' : attr.label_list(aspects = [dep_aspect])})",
        "",
        "dep_rule = rule(implementation = _dep_rule_impl,",
        "  attrs = {'deps' : attr.label_list(allow_files = True), 'output' : attr.output()})");

    // Target graph: test -> aspect_propagating_target -> alpha -> beta_output
    // beta_output is an output of target `beta`
    // beta -> charlie
    scratch.file(
        "test/BUILD",
        "load('//test:lib.bzl', 'root_rule', 'dep_with_aspect_rule', 'dep_rule')",
        "",
        "root_rule(name = 'test', deps = [':aspect_propagating_target'])",
        "dep_with_aspect_rule(name = 'aspect_propagating_target', deps = [':alpha'])",
        "dep_rule(name = 'alpha', deps = [':beta_output'])",
        "dep_rule(name = 'beta', deps = [':charlie'], output = 'beta_output')",
        "dep_rule(name = 'charlie')");
  }

  @Test
  public void testAspectOnAspectApplyToGeneratingRules_bothPropagate() throws Exception {
    setupAspectOnAspectTargetGraph(
        /* applyRootToGeneratingRules= */ true, /* applyDepToGeneratingRules= */ true);

    StarlarkProvider.Key rootInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "RootAspectInfo");
    StarlarkProvider.Key depInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "DepAspectInfo");

    AnalysisResult analysisResult = update("//test:test");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    StructImpl rootInfo = (StructImpl) target.get(rootInfoKey);
    StructImpl depInfo = (StructImpl) target.get(depInfoKey);

    assertThat(rootInfo.getValue("both_labels", Sequence.class))
        .containsExactly(
            Label.parseAbsolute("//test:alpha", ImmutableMap.of()),
            Label.parseAbsolute("//test:beta_output", ImmutableMap.of()),
            Label.parseAbsolute("//test:charlie", ImmutableMap.of()));
    assertThat(rootInfo.getValue("root_only_labels", Sequence.class)).isEmpty();
    assertThat(depInfo.getValue("labels", Sequence.class))
        .containsExactly(
            Label.parseAbsolute("//test:alpha", ImmutableMap.of()),
            Label.parseAbsolute("//test:beta", ImmutableMap.of()),
            Label.parseAbsolute("//test:charlie", ImmutableMap.of()));
  }

  @Test
  public void testAspectOnAspectApplyToGeneratingRules_neitherPropagate() throws Exception {
    setupAspectOnAspectTargetGraph(
        /* applyRootToGeneratingRules= */ false, /* applyDepToGeneratingRules= */ false);

    StarlarkProvider.Key rootInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "RootAspectInfo");
    StarlarkProvider.Key depInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "DepAspectInfo");

    AnalysisResult analysisResult = update("//test:test");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    StructImpl rootInfo = (StructImpl) target.get(rootInfoKey);
    StructImpl depInfo = (StructImpl) target.get(depInfoKey);

    assertThat(rootInfo.getValue("both_labels", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
    assertThat(rootInfo.getValue("root_only_labels", Sequence.class)).isEmpty();
    assertThat(depInfo.getValue("labels", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
  }

  @Test
  public void testAspectOnAspectApplyToGeneratingRules_rootOnly() throws Exception {
    setupAspectOnAspectTargetGraph(
        /* applyRootToGeneratingRules= */ true, /* applyDepToGeneratingRules= */ false);

    StarlarkProvider.Key rootInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "RootAspectInfo");
    StarlarkProvider.Key depInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "DepAspectInfo");

    AnalysisResult analysisResult = update("//test:test");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    StructImpl rootInfo = (StructImpl) target.get(rootInfoKey);
    StructImpl depInfo = (StructImpl) target.get(depInfoKey);

    assertThat(rootInfo.getValue("both_labels", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
    assertThat(rootInfo.getValue("root_only_labels", Sequence.class))
        .containsExactly(
            Label.parseAbsolute("//test:beta_output", ImmutableMap.of()),
            Label.parseAbsolute("//test:charlie", ImmutableMap.of()));
    assertThat(depInfo.getValue("labels", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
  }

  @Test
  public void testAspectOnAspectApplyToGeneratingRules_depOnly() throws Exception {
    setupAspectOnAspectTargetGraph(
        /* applyRootToGeneratingRules= */ false, /* applyDepToGeneratingRules= */ true);

    StarlarkProvider.Key rootInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "RootAspectInfo");
    StarlarkProvider.Key depInfoKey =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//test:lib.bzl", ImmutableMap.of()), "DepAspectInfo");

    AnalysisResult analysisResult = update("//test:test");
    ConfiguredTarget target = Iterables.getOnlyElement(analysisResult.getTargetsToBuild());
    StructImpl rootInfo = (StructImpl) target.get(rootInfoKey);
    StructImpl depInfo = (StructImpl) target.get(depInfoKey);

    assertThat(rootInfo.getValue("both_labels", Sequence.class))
        .containsExactly(Label.parseAbsolute("//test:alpha", ImmutableMap.of()));
    assertThat(rootInfo.getValue("root_only_labels", Sequence.class)).isEmpty();
    assertThat(depInfo.getValue("labels", Sequence.class))
        .containsExactly(
            Label.parseAbsolute("//test:alpha", ImmutableMap.of()),
            Label.parseAbsolute("//test:beta", ImmutableMap.of()),
            Label.parseAbsolute("//test:charlie", ImmutableMap.of()));
  }

  /** StarlarkAspectTest with "keep going" flag */
  @RunWith(JUnit4.class)
  public static final class WithKeepGoing extends StarlarkDefinedAspectsTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.KEEP_GOING);
    }

    @Override
    protected boolean keepGoing() {
      return true;
    }
  }
}
