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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCountAtLeast;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import com.google.devtools.build.lib.analysis.util.ExpectedTrimmedConfigurationErrors;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.OutputFilter.RegexOutputFilter;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.LoadingFailureEvent;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.NotifyingHelper.Listener;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@link BuildView}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class BuildViewTest extends BuildViewTestBase {
  private static final Function<AnalysisFailureEvent, Pair<String, String>>
      ANALYSIS_EVENT_TO_STRING_PAIR =
          new Function<AnalysisFailureEvent, Pair<String, String>>() {
    @Override
    public Pair<String, String> apply(AnalysisFailureEvent event) {
      return Pair.of(
          event.getFailedTarget().getLabel().toString(), event.getLegacyFailureReason().toString());
    }
  };

  @Test
  public void directoryArtifactInRoot() throws Exception {
    scratch.file(
        "BUILD", "genrule(name = 'slurps_dir', srcs = ['.'], outs = ['out'], cmd = 'touch $@')");
    // Expect no errors.
    update("//:slurps_dir");
  }

  @Test
  public void testRuleConfiguredTarget() throws Exception {
    scratch.file("pkg/BUILD",
        "genrule(name='foo', ",
        "        cmd = '',",
        "        srcs=['a.src'],",
        "        outs=['a.out'])");
    update("//pkg:foo");
    Rule ruleTarget = (Rule) getTarget("//pkg:foo");
    assertThat(ruleTarget.getRuleClass()).isEqualTo("genrule");

    ConfiguredTargetAndData ruleCTAT = getConfiguredTargetAndTarget("//pkg:foo");

    assertThat(ruleCTAT.getTarget()).isSameInstanceAs(ruleTarget);
  }

  @Test
  public void testFilterByTargets() throws Exception {
    scratch.file("tests/BUILD",
        "sh_test(name = 'small_test_1',",
        "        srcs = ['small_test_1.sh'],",
        "        data = [':xUnit'],",
        "        size = 'small',",
        "        tags = ['tag1'])",
        "",
        "sh_test(name = 'small_test_2',",
        "        srcs = ['small_test_2.sh'],",
        "        size = 'small',",
        "        tags = ['tag2'])",
        "",
        "",
        "test_suite( name = 'smallTests', tags=['small'])");
    //scratch.file("tests/small_test_1.py");

    update("//tests:smallTests");
    ConfiguredTargetAndData test1 = getConfiguredTargetAndTarget("//tests:small_test_1");
    ConfiguredTargetAndData test2 = getConfiguredTargetAndTarget("//tests:small_test_2");
    ConfiguredTargetAndData suite = getConfiguredTargetAndTarget("//tests:smallTests");

    ConfiguredTarget test1CT = test1.getConfiguredTarget();
    ConfiguredTarget test2CT = test2.getConfiguredTarget();
    ConfiguredTarget suiteCT = suite.getConfiguredTarget();
    assertNoEvents(); // start from a clean slate

    Collection<ConfiguredTarget> targets =
        new LinkedHashSet<>(ImmutableList.of(test1CT, test2CT, suiteCT));
    targets =
        Lists.<ConfiguredTarget>newArrayList(
            BuildView.filterTestsByTargets(
                targets,
                Sets.newHashSet(test1.getTarget().getLabel(), suite.getTarget().getLabel())));
    assertThat(targets).containsExactlyElementsIn(Sets.newHashSet(test1CT, suiteCT));
  }

  @Test
  public void testSourceArtifact() throws Exception {
    setupDummyRule();
    update("//pkg:a.src");
    InputFileConfiguredTarget inputCT = getInputFileConfiguredTarget("//pkg:a.src");
    Artifact inputArtifact = inputCT.getArtifact();
    assertThat(getGeneratingAction(inputArtifact)).isNull();
    assertThat(inputArtifact.getExecPathString()).isEqualTo("pkg/a.src");
  }

  @Test
  public void testGeneratedArtifact() throws Exception {
    setupDummyRule();
    update("//pkg:a.out");
    ConfiguredTargetAndData ctad = getConfiguredTargetAndData("//pkg:a.out");
    OutputFileConfiguredTarget output = (OutputFileConfiguredTarget) ctad.getConfiguredTarget();
    Artifact outputArtifact = output.getArtifact();
    assertThat(outputArtifact.getRoot())
        .isEqualTo(
            ctad.getConfiguration()
                .getBinDirectory(output.getLabel().getPackageIdentifier().getRepository()));
    assertThat(outputArtifact.getExecPath())
        .isEqualTo(ctad.getConfiguration().getBinFragment().getRelative("pkg/a.out"));
    assertThat(outputArtifact.getRootRelativePath()).isEqualTo(PathFragment.create("pkg/a.out"));

    Action action = getGeneratingAction(outputArtifact);
    assertThat(action.getClass()).isSameInstanceAs(FailAction.class);
  }

  @Test
  public void testGetArtifactOwnerInStarlark() throws Exception {
    scratch.file(
        "foo/rule.bzl",
        "def _impl(ctx):",
        "  f = ctx.actions.declare_file('rule_output')",
        "  print('f owner is ' + str(f.owner))",
        "  ctx.actions.write(",
        "    output = f,",
        "    content = 'foo',",
        "  )",
        "gen = rule(implementation = _impl)");
    scratch.file("foo/BUILD", "load(':rule.bzl', 'gen')", "gen(name = 'a')");

    update("//foo:a");
    assertContainsEvent("DEBUG /workspace/foo/rule.bzl:3:3: f owner is //foo:a");
  }

  @Test
  public void testSyntaxErrorInDepPackage() throws Exception {
    // Check that a loading error in a dependency is properly reported.
    scratch.file("a/BUILD",
        "genrule(name='x',",
        "        srcs = ['file.txt'],",
        "        outs = ['foo'],",
        "        cmd = 'echo')",
        "@");  // syntax error

    scratch.file("b/BUILD",
        "genrule(name= 'cc',",
        "        tools = ['//a:x'],",
        "        outs = ['bar'],",
        "        cmd = 'echo')");

    reporter.removeHandler(failFastHandler);
    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//b:cc");

    assertContainsEvent("invalid character: '@'");
    assertThat(result.hasError()).isTrue();
  }

  @Test
  public void testReportsAnalysisRootCauses() throws Exception {
    scratch.file("private/BUILD",
        "genrule(",
        "    name='private',",
        "    outs=['private.out'],",
        "    cmd='',",
        "    visibility=['//visibility:private'])");
    scratch.file("foo/BUILD",
        "genrule(",
        "    name='foo',",
        "    tools=[':bar'],",
        "    outs=['foo.out'],",
        "    cmd='')",
        "genrule(",
        "    name='bar',",
        "    tools=['//private'],",
        "    outs=['bar.out'],",
        "    cmd='')");

    reporter.removeHandler(failFastHandler);
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();

    assertThat(recorder.events).hasSize(1);
    AnalysisFailureEvent event = recorder.events.get(0);
    assertThat(event.getLegacyFailureReason().toString()).isEqualTo("//foo:bar");
    assertThat(event.getFailedTarget().getLabel().toString()).isEqualTo("//foo:foo");

    assertThat(recorder.causes).hasSize(1);
    AnalysisRootCauseEvent cause = recorder.causes.get(0);
    assertThat(cause.getLabel().toString()).isEqualTo("//foo:bar");
  }

  @Test
  public void testTestOnlyFailureReported() throws Exception {
    scratch.file(
        "foo/BUILD",
        "genrule(",
        "    name='foo',",
        "    tools=[':bar'],",
        "    outs=['foo.out'],",
        "    cmd='')",
        "genrule(",
        "    name='bar',",
        "    outs=['bar.out'],",
        "    testonly=1,",
        "    cmd='')");

    reporter.removeHandler(failFastHandler);
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();

    assertThat(recorder.events).hasSize(1);
    AnalysisFailureEvent event = recorder.events.get(0);
    assertThat(event.getLegacyFailureReason().toString()).isEqualTo("//foo:foo");
    assertThat(event.getFailedTarget().getLabel().toString()).isEqualTo("//foo:foo");

    assertThat(recorder.causes).hasSize(1);
    AnalysisRootCauseEvent cause = recorder.causes.get(0);
    assertThat(cause.getLabel().toString()).isEqualTo("//foo:foo");
  }

  @Test
  public void testAnalysisReportsDependencyCycle() throws Exception {
    scratch.file("foo/BUILD", "sh_library(name='foo',deps=['//bar'])");
    scratch.file("bar/BUILD", "sh_library(name='bar',deps=[':bar'])");

    reporter.removeHandler(failFastHandler);
    AnalysisFailureRecorder recorder = new AnalysisFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(1);
    AnalysisFailureEvent event = recorder.events.get(0);
    assertThat(event.getConfigurationId()).isNotEqualTo(NullConfiguration.INSTANCE.getEventId());
  }

  @Test
  public void testReportsLoadingRootCauses() throws Exception {
    // This test checks that two simultaneous errors are both reported:
    // - missing outs attribute,
    // - package referenced in tools does not exist
    scratch.file("pkg/BUILD",
        "genrule(name='foo',",
        "        tools=['//nopackage:missing'],",
        "        cmd='')");

    reporter.removeHandler(failFastHandler);
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    // Note: no need to run analysis for a loading failure.
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//pkg:foo");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events)
        .contains(
            new LoadingFailureEvent(
                Label.parseAbsolute("//pkg:foo", ImmutableMap.of()),
                Label.parseAbsolute("//nopackage:missing", ImmutableMap.of())));
    assertContainsEvent("missing value for mandatory attribute 'outs'");
    assertContainsEvent("no such package 'nopackage'");
    // Skyframe correctly reports the other root cause as the genrule itself (since it is
    // missing attributes).
    assertThat(recorder.events).hasSize(2);
    assertThat(recorder.events)
        .contains(
            new LoadingFailureEvent(
                Label.parseAbsolute("//pkg:foo", ImmutableMap.of()),
                Label.parseAbsolute("//pkg:foo", ImmutableMap.of())));
  }

  @Test
  public void testMultipleRootCauseReporting() throws Exception {
    scratch.file("gp/BUILD",
        "sh_library(name = 'gp', deps = ['//p:p'])");
    scratch.file("p/BUILD",
        "sh_library(name = 'p', deps = ['//c1:not', '//c2:not'])");
    scratch.file("c1/BUILD");
    scratch.file("c2/BUILD");
    reporter.removeHandler(failFastHandler);
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//gp");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events).hasSize(2);
    assertThat(recorder.events)
        .contains(
            new LoadingFailureEvent(
                Label.parseAbsolute("//gp", ImmutableMap.of()),
                Label.parseAbsolute("//c1:not", ImmutableMap.of())));
    assertThat(recorder.events)
        .contains(
            new LoadingFailureEvent(
                Label.parseAbsolute("//gp", ImmutableMap.of()),
                Label.parseAbsolute("//c2:not", ImmutableMap.of())));
  }

  /**
   * Regression test for: "Package group includes are broken"
   */
  @Test
  public void testTopLevelPackageGroup() throws Exception {
    scratch.file("tropical/BUILD",
        "package_group(name='guava', includes=[':mango'])",
        "package_group(name='mango')");

    // If the analysis phase results in an error, this will throw an exception
    update("//tropical:guava");

    // Check if the included package group also got analyzed
    assertThat(getConfiguredTarget("//tropical:mango", null)).isNotNull();
  }

  @Test
  public void testTopLevelInputFile() throws Exception {
    scratch.file("tropical/BUILD",
        "exports_files(['file.txt'])");
    update("//tropical:file.txt");
    assertThat(getConfiguredTarget("//tropical:file.txt", null)).isNotNull();
  }

  @Test
  public void testGetDirectPrerequisites() throws Exception {
    scratch.file(
        "package/BUILD",
        "filegroup(name='top', srcs=[':inner', 'file'])",
        "sh_binary(name='inner', srcs=['script.sh'])");
    update("//package:top");
    ConfiguredTarget top = getConfiguredTarget("//package:top", getTargetConfiguration());
    Iterable<ConfiguredTarget> targets = getView().getDirectPrerequisitesForTesting(
        reporter, top, getBuildConfigurationCollection());
    Iterable<Label> labels = Iterables.transform(targets, target -> target.getLabel());
    assertThat(labels)
        .containsExactly(
            Label.parseAbsolute("//package:inner", ImmutableMap.of()),
            Label.parseAbsolute("//package:file", ImmutableMap.of()));
  }

  @Test
  public void testGetDirectPrerequisiteDependencies() throws Exception {
    // Override the trimming transition to not distort the results.
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.overrideTrimmingTransitionFactoryForTesting((rule) -> NoTransition.INSTANCE);
    useRuleClassProvider(builder.build());

    update();

    scratch.file(
        "package/BUILD",
        "filegroup(name='top', srcs=[':inner', 'file'])",
        "sh_binary(name='inner', srcs=['script.sh'])");
    ConfiguredTarget top = Iterables.getOnlyElement(update("//package:top").getTargetsToBuild());
    Iterable<Dependency> targets =
        getView()
            .getDirectPrerequisiteDependenciesForTesting(
                reporter, top, getBuildConfigurationCollection(), /*toolchainContext=*/ null)
            .values();

    Dependency innerDependency =
        Dependency.withTransitionAndAspects(
            Label.parseAbsolute("//package:inner", ImmutableMap.of()),
            NoTransition.INSTANCE,
            AspectCollection.EMPTY);
    Dependency fileDependency =
        Dependency.withTransitionAndAspects(
            Label.parseAbsolute("//package:file", ImmutableMap.of()),
            NullTransition.INSTANCE,
            AspectCollection.EMPTY);

    assertThat(targets).containsExactly(innerDependency, fileDependency);
  }

  /**
   * Tests that the {@code --output directory name} option cannot be used on
   * the command line.
   */
  @Test
  public void testConfigurationShortName() throws Exception {
    // Check that output directory name is still the name, otherwise this test is not testing what
    // we expect.
    CoreOptions options = Options.getDefaults(CoreOptions.class);
    options.outputDirectoryName = "/home/wonkaw/wonka_chocolate/factory/out";
    assertWithMessage("The flag's name may have been changed; this test may need to be updated.")
        .that(options.asMap().get("output directory name"))
        .isEqualTo("/home/wonkaw/wonka_chocolate/factory/out");

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> useConfiguration("--output directory name=foo"));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --output directory name=foo");
  }

  @Test
  public void testFileTranslations() throws Exception {
    scratch.file("foo/file");
    scratch.file("foo/BUILD",
        "exports_files(['file'])");
    useConfiguration("--message_translations=//foo:file");
    scratch.file("bar/BUILD",
        "sh_library(name = 'bar')");
    update("//bar");
  }

  // Regression test: "output_filter broken (but in a different way)"
  @Test
  public void testOutputFilterSeeWarning() throws Exception {
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      // TODO(b/129599328): fix or justify disabling
      return;
    }
    runAnalysisWithOutputFilter(Pattern.compile(".*"));
    assertContainsEvent("please do not import '//java/a:A.java'");
  }

  // Regression test: "output_filter broken (but in a different way)"
  @Test
  public void testOutputFilter() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      // TODO(b/129599328): fix or justify disabling
      return;
    }
    runAnalysisWithOutputFilter(Pattern.compile("^//java/c"));
    assertNoEvents();
  }

  @Test
  public void testOutputFilterWithDebug() throws Exception {
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      // TODO(b/129599328): fix or justify disabling
      return;
    }
    scratch.file(
        "java/a/BUILD",
        "java_library(name = 'a',",
        "  srcs = ['A.java'],",
        "  deps = ['//java/b'])");
    scratch.file(
        "java/b/rules.bzl",
        "def _impl(ctx):",
        "  print('debug in b')",
        "  ctx.actions.write(",
        "    output = ctx.outputs.my_output,",
        "    content = 'foo',",
        "  )",
        "gen = rule(implementation = _impl, outputs = {'my_output': 'B.java'})");
    scratch.file(
        "java/b/BUILD",
        "load(':rules.bzl', 'gen')",
        "gen(name='src')",
        "java_library(name = 'b', srcs = [':src'])");
    reporter.setOutputFilter(RegexOutputFilter.forPattern(Pattern.compile("^//java/a")));

    update("//java/a:a");
    assertContainsEvent("DEBUG /workspace/java/b/rules.bzl:2:3: debug in b");
  }

  @Test
  public void testAnalysisErrorMessageWithKeepGoing() throws Exception {
    scratch.file("a/BUILD", "sh_binary(name='a', srcs=['a1.sh', 'a2.sh'])");
    reporter.removeHandler(failFastHandler);
    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//a");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("errors encountered while analyzing target '//a:a'");
  }

  /**
   * Regression test: Exception in ConfiguredTargetGraph.checkForCycles()
   * when multiple top-level targets depend on the same cycle.
   */
  @Test
  public void testCircularDependencyBelowTwoTargets() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    scratch.file("foo/BUILD",
        "sh_library(name = 'top1', srcs = ['top1.sh'], deps = [':rec1'])",
        "sh_library(name = 'top2', srcs = ['top2.sh'], deps = [':rec1'])",
        "sh_library(name = 'rec1', srcs = ['rec1.sh'], deps = [':rec2'])",
        "sh_library(name = 'rec2', srcs = ['rec2.sh'], deps = [':rec1'])"
    );
    reporter.removeHandler(failFastHandler);
    AnalysisResult result =
        update(defaultFlags().with(Flag.KEEP_GOING), "//foo:top1", "//foo:top2");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("in sh_library rule //foo:rec1: cycle in dependency graph:\n");
    assertContainsEvent("in sh_library rule //foo:top");
  }

  // Regression test: cycle node depends on error.
  @Test
  public void testErrorBelowCycle() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling (also b/67412276: handle cycles properly).
      return;
    }
    scratch.file("foo/BUILD",
        "sh_library(name = 'top', deps = ['mid'])",
        "sh_library(name = 'mid', deps = ['bad', 'cycle1'])",
        "sh_library(name = 'bad', srcs = ['//badbuild:isweird'])",
        "sh_library(name = 'cycle1', deps = ['cycle2', 'mid'])",
        "sh_library(name = 'cycle2', deps = ['cycle1'])");
    scratch.file("badbuild/BUILD", "");
    reporter.removeHandler(failFastHandler);
    injectGraphListenerForTesting(Listener.NULL_LISTENER, /*deterministic=*/ true);
    assertThrows(ViewCreationFailedException.class, () -> update("//foo:top"));
    assertContainsEvent("no such target '//badbuild:isweird': target 'isweird' not declared in "
        + "package 'badbuild'");
    assertContainsEvent("and referenced by '//foo:bad'");
    assertContainsEvent("in sh_library rule //foo");
    assertContainsEvent("cycle in dependency graph");
    assertEventCountAtLeast(2, eventCollector);
  }

  @Test
  public void testErrorBelowCycleKeepGoing() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    scratch.file("foo/BUILD",
        "sh_library(name = 'top', deps = ['mid'])",
        "sh_library(name = 'mid', deps = ['bad', 'cycle1'])",
        "sh_library(name = 'bad', srcs = ['//badbuild:isweird'])",
        "sh_library(name = 'cycle1', deps = ['cycle2', 'mid'])",
        "sh_library(name = 'cycle2', deps = ['cycle1'])");
    scratch.file("badbuild/BUILD", "");
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//foo:top");
    assertContainsEvent("no such target '//badbuild:isweird': target 'isweird' not declared in "
        + "package 'badbuild'");
    assertContainsEvent("and referenced by '//foo:bad'");
    assertContainsEvent("in sh_library rule //foo");
    assertContainsEvent("cycle in dependency graph");
    // This error is triggered both in configuration trimming (which visits the transitive target
    // closure) and in the normal configured target cycle detection path. So we get an additional
    // instance of this check (which varies depending on whether Skyframe loading phase is enabled).
    // TODO(gregce): Fix above and uncomment the below. Note that the duplicate doesn't make it into
    // real user output (it only affects tests).
    //  assertEventCount(3, eventCollector);
  }

  @Test
  public void testAnalysisEntryHasActionsEvenWithError() throws Exception {
    scratch.file("foo/BUILD",
        "cc_binary(name = 'foo', linkshared = 1, srcs = ['foo.cc'])");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//foo:foo"));
  }

  @Test
  public void testHelpfulErrorForWrongPackageLabels() throws Exception {
    reporter.removeHandler(failFastHandler);

    scratch.file("x/BUILD",
        "cc_library(name='x', srcs=['x.cc'])");
    scratch.file("y/BUILD",
        "cc_library(name='y', srcs=['y.cc'], deps=['//x:z'])");

    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//y:y");
    assertThat(result.hasError()).isTrue();
    assertContainsEvent("no such target '//x:z': "
        + "target 'z' not declared in package 'x' "
        + "defined by /workspace/x/BUILD and referenced by '//y:y'");
  }

  @Test
  public void testNewActionsAreDifferentAndDontConflict() throws Exception {
    scratch.file("pkg/BUILD",
        "genrule(name='a', ",
        "        cmd='',",
        "        outs=['a.out'])");
    OutputFileConfiguredTarget outputCT = (OutputFileConfiguredTarget)
        Iterables.getOnlyElement(update("//pkg:a.out").getTargetsToBuild());
    Artifact outputArtifact = outputCT.getArtifact();
    Action action = getGeneratingAction(outputArtifact);
    assertThat(action).isNotNull();
    scratch.overwriteFile("pkg/BUILD",
        "genrule(name='a', ",
        "        cmd='false',",
        "        outs=['a.out'])");
    update("//pkg:a.out");
    assertWithMessage("Actions should not be compatible")
        .that(Actions.canBeShared(actionKeyContext, action, getGeneratingAction(outputArtifact)))
        .isFalse();
  }

  /**
   * This test exercises the case where we invalidate (mark dirty) a node in one build command
   * invocation and the revalidation (because it did not change) happens in a subsequent build
   * command call.
   *
   * - In the first update call we construct A.
   *
   * - Then we construct B and we make the glob get invalidated. We do that by deleting a file
   * because it depends on the directory listing. Because of that A gets invalidated.
   *
   * - Then we construct A again. The glob gets revalidated because it is still matching just A.java
   * and A configured target gets revalidated too. At the end of the analysis A java action should
   * be in the action graph.
   */
  @Test
  public void testMultiBuildInvalidationRevalidation() throws Exception {
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      // TODO(b/129599328): fix or justify disabling
      return;
    }
    scratch.file("java/a/A.java", "bla1");
    scratch.file("java/a/C.java", "bla2");
    scratch.file("java/a/BUILD",
        "java_test(name = 'A',",
        "          srcs = glob(['A*.java']))",
        "java_test(name = 'B',",
        "          srcs = ['B.java'])");
    ConfiguredTarget ct = Iterables.getOnlyElement(update("//java/a:A").getTargetsToBuild());
    scratch.deleteFile("java/a/C.java");
    update("//java/a:B");
    update("//java/a:A");
    assertThat(getGeneratingAction(getBinArtifact("A_deploy.jar", ct))).isNotNull();
  }

  /**
   * Regression test: ClassCastException in SkyframeLabelVisitor.updateRootCauses.
   */
  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitivelyBadTarget() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("parent/BUILD",
        "sh_library(name = 'foo',",
        "           srcs = ['//badpkg1:okay-target', '//okaypkg:transitively-bad-target'])");
    Path badpkg1BuildFile =
        scratch.file("badpkg1/BUILD", "exports_files(['okay-target'])", "fail()");
    scratch.file("okaypkg/BUILD",
        "sh_library(name = 'transitively-bad-target',",
        "           srcs = ['//badpkg2:bad-target'])");
    Path badpkg2BuildFile =
        scratch.file("badpkg2/BUILD", "sh_library(name = 'bad-target')", "fail()");
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:foo");
    assertThat(getFrequencyOfErrorsWithLocation(badpkg1BuildFile.asFragment(), eventCollector))
        .isEqualTo(1);
    assertThat(getFrequencyOfErrorsWithLocation(badpkg2BuildFile.asFragment(), eventCollector))
        .isEqualTo(1);
  }

  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitiveCycle_NotIncremental() throws Exception {
    runTestDepOnGoodTargetInBadPkgAndTransitiveCycle(/*incremental=*/false);
  }

  @Test
  public void testDepOnGoodTargetInBadPkgAndTransitiveCycle_Incremental() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    runTestDepOnGoodTargetInBadPkgAndTransitiveCycle(/*incremental=*/true);
  }

  /**
   * Regression test: in keep_going mode, cycles in target graph are reported even if the package is
   * in error.
   */
  @Test
  public void testCycleReporting_TargetCycleWhenPackageInError() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "cycles/BUILD",
        "sh_library(name = 'a', deps = [':b'])",
        "sh_library(name = 'b', deps = [':a'])",
        "x = 1//0"); // dynamic error
    update(defaultFlags().with(Flag.KEEP_GOING), "//cycles:a");
    assertContainsEvent("division by zero");
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testTransitiveLoadingDoesntShortCircuitInKeepGoing() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "parent/BUILD", "sh_library(name = 'a', deps = ['//child:b'])", "fail('parentisbad')");
    scratch.file("child/BUILD", "sh_library(name = 'b')", "fail('childisbad')");
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:a");
    assertContainsEventWithFrequency("parentisbad", 1);
    assertContainsEventWithFrequency("childisbad", 1);
    assertContainsEventWithFrequency("and referenced by '//parent:a'", 1);
  }

  /**
   * Smoke test for the Skyframe code path.
   */
  @Test
  public void testSkyframe() throws Exception {
    setupDummyRule();
    String aoutLabel = "//pkg:a.out";

    update(aoutLabel);

    // However, a ConfiguredTarget was actually produced.
    ConfiguredTarget target = Iterables.getOnlyElement(getAnalysisResult().getTargetsToBuild());
    assertThat(target.getLabel().toString()).isEqualTo(aoutLabel);

    Artifact aout = target.getProvider(FileProvider.class).getFilesToBuild().getSingleton();
    Action action = getGeneratingAction(aout);
    assertThat(action.getClass()).isSameInstanceAs(FailAction.class);
  }

  /**
   * ConfiguredTargetFunction should not register actions in legacy Blaze ActionGraph unless
   * the creation of the node is successful.
   */
  @Test
  public void testActionsNotRegisteredInLegacyWhenError() throws Exception {
    // First find the artifact we want to make sure is not generated by an action with an error.
    // Then update the BUILD file and re-analyze.
    scratch.file("actions_not_registered/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    ConfiguredTarget foo =
        Iterables.getOnlyElement(update("//actions_not_registered:foo").getTargetsToBuild());
    Artifact fooOut = foo.getProvider(FileProvider.class).getFilesToBuild().getSingleton();
    assertThat(getActionGraph().getGeneratingAction(fooOut)).isNotNull();
    clearAnalysisResult();

    scratch.overwriteFile("actions_not_registered/BUILD",
        "cc_binary(name = 'foo', linkshared = 1, srcs = ['foo.cc'])");

    reporter.removeHandler(failFastHandler);

    assertThrows(ViewCreationFailedException.class, () -> update("//actions_not_registered:foo"));
    assertThat(getActionGraph().getGeneratingAction(fooOut)).isNull();
  }

  /**
   * Regression test:
   * "skyframe: ArtifactFactory and ConfiguredTargets out of sync".
   */
  @Test
  public void testSkyframeAnalyzeRuleThenItsOutputFile() throws Exception {
    scratch.file("pkg/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])");

    scratch.file("pkg2/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])");
    String aoutLabel = "//pkg:a.out";

    update("//pkg2:foo");
    update("//pkg:foo");
    scratch.overwriteFile("pkg2/BUILD",
        "testing_dummy_rule(name='foo', ",
        "                   srcs=['a.src'],",
        "                   outs=['a.out'])",
        "# Comment");

    update("//pkg:a.out");

    // However, a ConfiguredTarget was actually produced.
    ConfiguredTarget target = Iterables.getOnlyElement(getAnalysisResult().getTargetsToBuild());
    assertThat(target.getLabel().toString()).isEqualTo(aoutLabel);

    Artifact aout = target.getProvider(FileProvider.class).getFilesToBuild().getSingleton();
    Action action = getGeneratingAction(aout);
    assertThat(action.getClass()).isSameInstanceAs(FailAction.class);
  }

  /**
   * Tests that skyframe reports the root cause as being the target that depended on the symlink
   * cycle.
   */
  @Test
  public void testRootCauseReportingFileSymlinks() throws Exception {
    scratch.file("gp/BUILD",
        "sh_library(name = 'gp', deps = ['//p'])");
    scratch.file("p/BUILD",
        "sh_library(name = 'p', deps = ['//c'])");
    scratch.file("c/BUILD",
        "sh_library(name = 'c', deps = [':c1', ':c2'])",
        "sh_library(name = 'c1', deps = ['//cycles1'])",
        "sh_library(name = 'c2', deps = ['//cycles2'])");
    Path cycles1BuildFilePath = scratch.file("cycles1/BUILD",
        "sh_library(name = 'cycles1', srcs = glob(['*.sh']))");
    Path cycles2BuildFilePath = scratch.file("cycles2/BUILD",
        "sh_library(name = 'cycles2', srcs = glob(['*.sh']))");
    cycles1BuildFilePath.getParentDirectory().getRelative("cycles1.sh").createSymbolicLink(
        PathFragment.create("cycles1.sh"));
    cycles2BuildFilePath.getParentDirectory().getRelative("cycles2.sh").createSymbolicLink(
        PathFragment.create("cycles2.sh"));
    reporter.removeHandler(failFastHandler);
    LoadingFailureRecorder recorder = new LoadingFailureRecorder();
    eventBus.register(recorder);
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//gp");
    assertThat(result.hasError()).isTrue();
    assertThat(recorder.events)
        .containsExactly(
            new LoadingFailureEvent(
                Label.parseAbsolute("//gp", ImmutableMap.of()),
                Label.parseAbsolute("//cycles1", ImmutableMap.of())),
            new LoadingFailureEvent(
                Label.parseAbsolute("//gp", ImmutableMap.of()),
                Label.parseAbsolute("//cycles2", ImmutableMap.of())));
  }

  /**
   * Regression test for bug when a configured target has missing deps, but also depends
   * transitively on an error. We build //foo:query, which depends on a valid and an invalid target
   * pattern. We first make sure the invalid target pattern is in the graph, so that it throws when
   * requested by //foo:query. Then, when bubbling the invalid target pattern error up, //foo:query
   * must cope with the combination of an error and a missing dep.
   */
  @Test
  public void testGenQueryWithBadTargetAndUnfinishedTarget() throws Exception {
    // The target //foo:zquery is used to force evaluation of //foo:nosuchtarget before the target
    // patterns in //foo:query are enqueued for evaluation. That way, //foo:query will depend on one
    // invalid target pattern and two target patterns that haven't been evaluated yet.
    // It is important that a missing target pattern is requested before the exception is thrown, so
    // we have both //foo:b and //foo:z missing from the deps, in the hopes that at least one of
    // them will come before //foo:nosuchtarget.
    scratch.file(
        "foo/BUILD",
        "genquery(name = 'query',",
        "         expression = 'deps(//foo:b) except //foo:nosuchtarget except //foo:z',",
        "         scope = ['//foo:a'])",
        "genquery(name = 'zquery',",
        "         expression = 'deps(//foo:nosuchtarget)',",
        "         scope = ['//foo:a'])",
        "sh_library(name = 'a')",
        "sh_library(name = 'b')",
        "sh_library(name = 'z')");
    reporter.removeHandler(failFastHandler);
    ViewCreationFailedException e =
        assertThrows(ViewCreationFailedException.class, () -> update("//foo:zquery"));
    assertThat(e)
        .hasMessageThat()
        .contains("Analysis of target '//foo:zquery' failed; build aborted");
    e = assertThrows(ViewCreationFailedException.class, () -> update("//foo:query"));
    assertThat(e)
        .hasMessageThat()
        .contains("Analysis of target '//foo:query' failed; build aborted");
  }

  /**
   * Tests that rules with configurable attributes can be accessed through {@link
   * com.google.devtools.build.lib.skyframe.PostConfiguredTargetFunction}.
   * This is a regression test for a Bazel crash.
   */
  @Test
  public void testPostProcessedConfigurableAttributes() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    useConfiguration("--cpu=k8");
    reporter.removeHandler(failFastHandler); // Expect errors from action conflicts.
    scratch.file(
        "conflict/BUILD",
        "config_setting(name = 'a', values = {'test_arg': 'a'})",
        "cc_library(name='x', srcs=select({':a': ['a.cc'], '//conditions:default': ['foo.cc']}))",
        "cc_binary(name='_objs/x/foo.o', srcs=['bar.cc'])");
    AnalysisResult result =
        update(defaultFlags().with(Flag.KEEP_GOING), "//conflict:_objs/x/foo.o", "//conflict:x");
    assertThat(result.hasError()).isTrue();
    // Expect to reach this line without a Precondition-triggered NullPointerException.
    assertContainsEvent("file 'conflict/_objs/x/foo.o' is generated by these conflicting actions");
  }

  @Test
  public void testCycleDueToJavaLauncherConfiguration() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      // Trimmed configurations don't yet support late-bound attributes.
      // TODO(gregce): re-enable this when ready.
      return;
    }
    scratch.file("foo/BUILD",
        "java_binary(name = 'java', srcs = ['DoesntMatter.java'])",
        "cc_binary(name = 'cpp', data = [':java'])");
    // Everything is fine - the dependency graph is acyclic.
    update("//foo:java", "//foo:cpp");
    if (getTargetConfiguration().trimConfigurations()) {
      fail(ExpectedTrimmedConfigurationErrors.LATE_BOUND_ATTRIBUTES_UNSUPPORTED);
    }
    // Now there will be an analysis-phase cycle because the java_binary now has an implicit dep on
    // the cc_binary launcher.
    useConfiguration("--java_launcher=//foo:cpp");
    reporter.removeHandler(failFastHandler);
    ViewCreationFailedException expected =
        assertThrows(ViewCreationFailedException.class, () -> update("//foo:java", "//foo:cpp"));
    assertThat(expected)
        .hasMessageThat()
        .matches("Analysis of target '//foo:(java|cpp)' failed; build aborted.*");
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testDependsOnBrokenTarget() throws Exception {
    scratch.file("foo/BUILD",
        "sh_test(name = 'test', srcs = ['test.sh'], data = ['//bar:data'])");
    scratch.file("bar/BUILD",
        "BROKEN BROKEN BROKEN!!!");
    reporter.removeHandler(failFastHandler);
    ViewCreationFailedException expected =
        assertThrows(ViewCreationFailedException.class, () -> update("//foo:test"));
    assertThat(expected)
        .hasMessageThat()
        .matches("Analysis of target '//foo:test' failed; build aborted.*");
  }

  /**
   * Regression test: IllegalStateException in BuildView.update() on circular dependency instead of
   * graceful failure.
   */
  @Test
  public void testCircularDependency() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    scratch.file("cycle/BUILD",
        "cc_library(name = 'foo', srcs = ['foo.cc'], deps = [':bar'])",
        "cc_library(name = 'bar', srcs = ['bar.cc'], deps = [':foo'])");
    reporter.removeHandler(failFastHandler);
    ViewCreationFailedException expected =
        assertThrows(ViewCreationFailedException.class, () -> update("//cycle:foo"));
    assertContainsEvent("in cc_library rule //cycle:foo: cycle in dependency graph:");
    assertThat(expected)
        .hasMessageThat()
        .contains("Analysis of target '//cycle:foo' failed; build aborted");
  }

  /**
   * Regression test: IllegalStateException in BuildView.update() on circular dependency instead of
   * graceful failure.
   */
  @Test
  public void testCircularDependencyWithKeepGoing() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67412276): handle cycles properly.
      return;
    }
    scratch.file("cycle/BUILD",
        "cc_library(name = 'foo', srcs = ['foo.cc'], deps = [':bar'])",
        "cc_library(name = 'bar', srcs = ['bar.cc'], deps = [':foo'])",
        "cc_library(name = 'bat', srcs = ['bat.cc'], deps = [':bas'])",
        "cc_library(name = 'bas', srcs = ['bas.cc'], deps = [':bau'])",
        "cc_library(name = 'bau', srcs = ['bas.cc'], deps = [':bas'])",
        "cc_library(name = 'baz', srcs = ['baz.cc'])");
    reporter.removeHandler(failFastHandler);
    LoadingFailureRecorder loadingFailureRecorder = new LoadingFailureRecorder();
    AnalysisFailureRecorder analysisFailureRecorder = new AnalysisFailureRecorder();
    eventBus.register(loadingFailureRecorder);
    eventBus.register(analysisFailureRecorder);
    update(eventBus, defaultFlags().with(Flag.KEEP_GOING),
        "//cycle:foo", "//cycle:bat", "//cycle:baz");
    assertContainsEvent("in cc_library rule //cycle:foo: cycle in dependency graph:");
    assertContainsEvent("in cc_library rule //cycle:bas: cycle in dependency graph:");
    assertContainsEvent(
        "errors encountered while analyzing target '//cycle:foo': it will not be built");
    assertContainsEvent(
        "errors encountered while analyzing target '//cycle:bat': it will not be built");
    // With interleaved loading and analysis, we can no longer distinguish loading-phase cycles
    // and analysis-phase cycles. This was previously reported as a loading-phase cycle, as it
    // happens with any configuration (cycle is hard-coded in the BUILD files). Also see the
    // test below.
    assertThat(Iterables.transform(analysisFailureRecorder.events, ANALYSIS_EVENT_TO_STRING_PAIR))
        .containsExactly(
            Pair.of("//cycle:foo", "//cycle:foo"), Pair.of("//cycle:bat", "//cycle:bas"));
  }

  @Test
  public void testLoadingErrorReportedCorrectly() throws Exception {
    scratch.file("a/BUILD", "cc_library(name='a')");
    scratch.file("b/BUILD", "cc_library(name='b', deps = ['//missing:lib'])");

    reporter.removeHandler(failFastHandler);
    AnalysisResult result = update(defaultFlags().with(Flag.KEEP_GOING), "//a", "//b");
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError())
        .contains("command succeeded, but there were loading phase errors");
  }

  @Test
  public void testVisibilityReferencesNonexistentPackage() throws Exception {
    scratch.file("z/a/BUILD",
        "py_library(name='a', visibility=['//nonexistent:nothing'])");
    scratch.file("z/b/BUILD",
        "py_library(name='b', deps=['//z/a:a'])");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//z/b:b"));
    assertContainsEvent("no such package 'nonexistent'");
  }

  // regression test ("java.lang.IllegalStateException: cannot happen")
  @Test
  public void testDefaultVisibilityInNonexistentPackage() throws Exception {
    scratch.file("z/a/BUILD",
        "package(default_visibility=['//b'])",
        "py_library(name='alib')");
    scratch.file("z/b/BUILD",
        "py_library(name='b', deps=['//z/a:alib'])");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//z/b:b"));
    assertContainsEvent("no such package 'b'");
  }

  @Test
  public void testNonTopLevelErrorsPrintedExactlyOnce() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    scratch.file("parent/BUILD",
        "sh_library(name = 'a', deps = ['//child:b'])");
    scratch.file("child/BUILD", "sh_library(name = 'b')", "fail('some error')");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//parent:a"));
    assertContainsEventWithFrequency("some error", 1);
    assertContainsEventWithFrequency(
        "Target '//child:b' contains an error and its package is in error and referenced "
        + "by '//parent:a'", 1);
  }

  @Test
  public void testNonTopLevelErrorsPrintedExactlyOnce_KeepGoing() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    scratch.file("parent/BUILD",
        "sh_library(name = 'a', deps = ['//child:b'])");
    scratch.file("child/BUILD", "sh_library(name = 'b')", "fail('some error')");
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:a");
    assertContainsEventWithFrequency("some error", 1);
    assertContainsEventWithFrequency(
        "Target '//child:b' contains an error and its package is in error and referenced "
        + "by '//parent:a'", 1);
  }

  @Test
  public void testNonTopLevelErrorsPrintedExactlyOnce_ActionListener() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    scratch.file("parent/BUILD",
        "sh_library(name = 'a', deps = ['//child:b'])");
    scratch.file("child/BUILD", "sh_library(name = 'b')", "fail('some error')");
    scratch.file("okay/BUILD",
        "sh_binary(name = 'okay', srcs = ['okay.sh'])");
    useConfiguration("--experimental_action_listener=//parent:a");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//okay"));
    assertContainsEventWithFrequency("some error", 1);
    assertContainsEventWithFrequency(
        "Target '//child:b' contains an error and its package is in error and referenced "
        + "by '//parent:a'", 1);
  }

  @Test
  public void testNonTopLevelErrorsPrintedExactlyOnce_ActionListener_KeepGoing() throws Exception {
    if (getInternalTestExecutionMode() != TestConstants.InternalTestExecutionMode.NORMAL) {
      // TODO(b/67651960): fix or justify disabling.
      return;
    }
    scratch.file("parent/BUILD",
        "sh_library(name = 'a', deps = ['//child:b'])");
    scratch.file("child/BUILD", "sh_library(name = 'b')", "fail('some error')");
    scratch.file("okay/BUILD",
        "sh_binary(name = 'okay', srcs = ['okay.sh'])");
    useConfiguration("--experimental_action_listener=//parent:a");
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//okay");
    assertContainsEventWithFrequency("some error", 1);
    assertContainsEventWithFrequency(
        "Target '//child:b' contains an error and its package is in error and referenced "
        + "by '//parent:a'", 1);
  }

  @Test
  public void testTopLevelTargetsAreTrimmedWithTrimmedConfigurations() throws Exception {
    scratch.file("foo/BUILD",
        "sh_library(name='x', ",
        "        srcs=['x.sh'])");
    useConfiguration("--experimental_dynamic_configs=on");
    AnalysisResult res = update("//foo:x");
    ConfiguredTarget topLevelTarget = Iterables.getOnlyElement(res.getTargetsToBuild());
    assertThat(getConfiguration(topLevelTarget).getFragmentsMap().keySet())
        .containsExactlyElementsIn(ruleClassProvider.getUniversalFragments());
  }

  /**
   * Here, injecting_rule injects an aspect which acts on a action_rule() and registers an action.
   * The action_rule() registers another action of its own.
   *
   * <p>This test asserts that both actions are reported.
   */
  @Test
  public void ruleExtraActionsDontHideAspectExtraActions() throws Exception {
    useConfiguration("--experimental_action_listener=//pkg:listener");

    scratch.file(
        "x/BUILD",
        "load(':extension.bzl', 'injecting_rule', 'action_rule')",
        "injecting_rule(name='a', deps=[':b'])",
        "action_rule(name='b')");

    scratch.file(
        "x/extension.bzl",
        "def _aspect1_impl(target, ctx):",
        "  ctx.actions.do_nothing(mnemonic='Mnemonic')",
        "  return []",
        "aspect1 = aspect(_aspect1_impl, attr_aspects=['deps'])",
        "",
        "def _injecting_rule_impl(ctx):",
        "  return []",
        "injecting_rule = rule(_injecting_rule_impl, ",
        "    attrs = { 'deps' : attr.label_list(aspects = [aspect1]) })",
        "",
        "def _action_rule_impl(ctx):",
        "  out = ctx.actions.declare_file(ctx.label.name)",
        "  ctx.actions.run_shell(outputs = [out], command = 'dontcare', mnemonic='Mnemonic')",
        "  return []",
        "action_rule = rule(_action_rule_impl, attrs = { 'deps' : attr.label_list() })");

    scratch.file(
        "pkg/BUILD",
        "extra_action(name='xa', cmd='echo dont-care')",
        "action_listener(name='listener', mnemonics=['Mnemonic'], extra_actions=[':xa'])");

    AnalysisResult analysisResult = update("//x:a");

    List<String> owners = new ArrayList<>();
    for (Artifact artifact : analysisResult.getTopLevelArtifactsToOwnerLabels().getArtifacts()) {
      if ("xa".equals(artifact.getExtension())) {
        owners.add(artifact.getOwnerLabel().toString());
      }
    }
    assertThat(owners).containsExactly("//x:b", "//x:b");
  }

  @Test
  public void testErrorMessageForMissingPackageGroup() throws Exception {
    scratch.file(
        "apple/BUILD",
        "py_library(name='apple', visibility=['//non:existent'])");
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//apple"));
    assertDoesNotContainEvent("implicitly depends upon");
  }

  @Test
  public void allowedRuleClassesAndAllowedRuleClassesWithWarning() throws Exception {
    setRulesAvailableInTests(
        (MockRule) () -> MockRule.define(
            "custom_rule",
            attr("deps", BuildType.LABEL_LIST)
                .allowedFileTypes()
                .allowedRuleClasses("java_library", "java_binary")
                .allowedRuleClassesWithWarning("genrule")));

    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'genlib',",
        "    srcs = [],",
        "    outs = ['genlib.out'],",
        "    cmd = 'echo hi > $@')",
        "custom_rule(",
        "    name = 'foo',",
        "    deps = [':genlib'])");

    update("//foo");
    assertContainsEvent(
        "WARNING /workspace/foo/BUILD:6:1: in deps attribute of custom_rule rule "
            + "//foo:foo: genrule rule '//foo:genlib' is unexpected here (expected java_library or "
            + "java_binary); continuing anyway");
  }

  @Test
  public void errorInImplicitDeps() throws Exception {
    setRulesAvailableInTests(
        (MockRule)
            () -> {
              try {
                return MockRule.define(
                    "custom_rule",
                    attr("$implicit1", BuildType.LABEL_LIST)
                        .defaultValue(
                            ImmutableList.of(
                                Label.parseAbsoluteUnchecked("//bad2:label"),
                                Label.parseAbsoluteUnchecked("//foo:dep"))),
                    attr("$implicit2", BuildType.LABEL)
                        .defaultValue(Label.parseAbsoluteUnchecked("//bad:label")));
              } catch (Type.ConversionException e) {
                throw new IllegalStateException(e);
              }
            });
    scratch.file("foo/BUILD", "custom_rule(name = 'foo')", "sh_library(name = 'dep')");
    scratch.file(
        "bad/BUILD",
        "sh_library(name = 'other_label', nonexistent_attribute = 'blah')",
        "sh_library(name = 'label')");
    // bad2/BUILD is completely missing.
    reporter.removeHandler(failFastHandler);
    update(defaultFlags().with(Flag.KEEP_GOING), "//foo:foo");
    assertContainsEvent(
        "every rule of type custom_rule implicitly depends upon the target '//bad2:label', but"
            + " this target could not be found because of: no such package 'bad2': BUILD file not"
            + " found");
    assertContainsEvent(
        "every rule of type custom_rule implicitly depends upon the target '//bad:label', but this "
            + "target could not be found because of: Target '//bad:label' contains an error and its"
            + " package is in error");
  }

  @Test
  public void onlyAllowedRuleClassesWithWarning() throws Exception {
    setRulesAvailableInTests(
        (MockRule) () -> MockRule.define(
            "custom_rule",
            attr("deps", BuildType.LABEL_LIST)
                .allowedFileTypes()
                .allowedRuleClassesWithWarning("genrule")));

    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'genlib',",
        "    srcs = [],",
        "    outs = ['genlib.out'],",
        "    cmd = 'echo hi > $@')",
        "custom_rule(",
        "    name = 'foo',",
        "    deps = [':genlib'])");

    update("//foo");
    assertContainsEvent(
        "WARNING /workspace/foo/BUILD:6:1: in deps attribute of custom_rule rule "
            + "//foo:foo: genrule rule '//foo:genlib' is unexpected here; continuing anyway");
  }

  @Test
  public void testExistingRule() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "genrule(name='foo', ",
        "        cmd = '',",
        "        srcs=['a.src'],",
        "        outs=['a.out'])",
        "print(existing_rule('foo')['kind'])",
        "print(existing_rule('bar'))");
    reporter.setOutputFilter(RegexOutputFilter.forPattern(Pattern.compile("^//pkg")));
    update("//pkg:foo");
    assertContainsEvent("DEBUG /workspace/pkg/BUILD:5:1: genrule");
    assertContainsEvent("DEBUG /workspace/pkg/BUILD:6:1: None");
  }

  @Test
  public void testExistingRules() throws Exception {
    scratch.file(
        "pkg/BUILD",
        "genrule(name='foo', ",
        "        cmd = '',",
        "        srcs=['a.src'],",
        "        outs=['a.out'])",
        "print(existing_rules().keys())");
    reporter.setOutputFilter(RegexOutputFilter.forPattern(Pattern.compile("^//pkg")));
    update("//pkg:foo");
    assertContainsEvent("DEBUG /workspace/pkg/BUILD:5:1: [\"foo\"]");
  }

  /** Runs the same test with trimmed configurations. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithTrimmedConfigurations extends BuildViewTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.TRIMMED_CONFIGURATIONS);
    }
  }

  /** Runs the same test with the Skyframe-based analysis prep. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithSkyframePrepareAnalysis extends BuildViewTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.SKYFRAME_PREPARE_ANALYSIS);
    }
  }

  /** Runs the same test with the Skyframe-based analysis prep and trimmed configurations. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithSkyframePrepareAnalysisAndTrimmedConfigurations extends BuildViewTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags()
          .with(Flag.SKYFRAME_PREPARE_ANALYSIS)
          .with(Flag.TRIMMED_CONFIGURATIONS);
    }

    // We can't recover from dependency cycles in TransitiveTargetFunction, so we disable the tests
    // for now. We will likely have to fix this before we can enable the new code by default.
    @Override
    @Test
    public void testCircularDependency() {
    }

    @Override
    @Test
    public void testErrorBelowCycleKeepGoing() {
    }

    @Override
    @Test
    public void testCircularDependencyBelowTwoTargets() {
    }

    @Override
    @Test
    public void testCycleReporting_TargetCycleWhenPackageInError() {
    }

    @Override
    @Test
    public void testCircularDependencyWithKeepGoing() {
    }

    @Override
    @Test
    public void testErrorBelowCycle() {
    }

    @Override
    @Test
    public void testAnalysisReportsDependencyCycle() {}
  }
}
