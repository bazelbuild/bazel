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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.fail;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.UncheckedActionConflictException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CppHelper;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.List;

/**
 * Unit tests for the {@link CompilationHelper} class.
 */
@RunWith(JUnit4.class)
public class CompilationHelperTest extends BuildViewTestCase {
  private AnalysisTestUtil.CollectingAnalysisEnvironment analysisEnvironment;

  @Before
  public final void createAnalysisEnvironment() throws Exception  {
    analysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
  }

  private List<Artifact> getAggregatingMiddleman(
      ConfiguredTarget rule, BuildConfiguration configuration, boolean withSolib) throws Exception {
    return CppHelper.getAggregatingMiddlemanForTesting(
        getRuleContext(rule, analysisEnvironment),
        ActionsTestUtil.NULL_ACTION_OWNER,
        "middleman",
        rule.getProvider(FileProvider.class).getFilesToBuild(),
        withSolib,
        configuration);
  }

  private List<Artifact> getAggregatingMiddleman(ConfiguredTarget rule, boolean withSolib)
      throws Exception {
    return getAggregatingMiddleman(rule, rule.getConfiguration(), withSolib);
  }

  /**
   * Tests that duplicate calls to
   * {@link com.google.devtools.build.lib.analysis.CompilationHelper#getAggregatingMiddleman}
   * with identical parameters return the same artifact.
   */
  @Test
  public void testDuplicateCallsReturnSameObject() throws Exception {
    ConfiguredTarget rule =
        scratchConfiguredTarget("package", "a", "cc_binary(name = 'a'," + "    srcs = ['a.cc'])");
    List<Artifact> middleman1 = getAggregatingMiddleman(rule, false);
    assertThat(middleman1).hasSize(1);
    List<Artifact> middleman2 = getAggregatingMiddleman(rule, false);
    assertThat(middleman2).hasSize(1);
    assertEquals(middleman1.get(0), middleman2.get(0));
  }

  /**
   * Tests that
   * {@link com.google.devtools.build.lib.analysis.CompilationHelper#getAggregatingMiddleman}
   * returns distinct artifacts even when called with identical rules, depending on
   * whether solib symlink are created.
   */
  @Test
  public void testMiddlemanAndSolibMiddlemanAreDistinct() throws Exception {
    ConfiguredTarget rule = scratchConfiguredTarget("package", "liba.so",
        "cc_binary(name = 'liba.so', srcs = ['a.cc'], linkshared = 1)");

    List<Artifact> middleman = getAggregatingMiddleman(rule, false);
    assertThat(middleman).hasSize(1);
    List<Artifact> middlemanWithSymlinks = getAggregatingMiddleman(rule, true);
    assertThat(middlemanWithSymlinks).hasSize(1);
    assertNotSame(middleman.get(0), middlemanWithSymlinks.get(0));
  }

  /**
   * Regression test: tests that Python CPU configurations are taken into account
   * when generating a rule's aggregating middleman, so that otherwise equivalent rules can sustain
   * distinct middlemen.
   */
  @Test
  public void testPythonCcConfigurations() throws Exception {
    setupJavaPythonCcConfigurationFiles();

    // Equivalent cc / Python configurations:

    ConfiguredTarget ccRuleA = getConfiguredTarget("//foo:liba.so");
    List<Artifact> middleman1 = getAggregatingMiddleman(ccRuleA, true);
    try {
      ConfiguredTarget ccRuleB = getConfiguredTarget("//foo:libb.so");
      getAggregatingMiddleman(ccRuleB, true);
      analysisEnvironment.registerWith(getMutableActionGraph());
      fail("Expected ActionConflictException due to same middleman artifact with different files");
    } catch (UncheckedActionConflictException e) {
      // Expected failure: same "purpose" and root directory sent to the middleman generator
      // (which results in the same output artifact), but different rules / middleman inputs.
    }

    // This should succeed because the py_binary's middleman is under the Python configuration's
    // internal directory, while the cc_binary's middleman is under the cc config's directory,
    // and both configurations are the same.
    ConfiguredTarget pyRuleB = getDirectPrerequisite(
        getConfiguredTarget("//foo:c"), "//foo:libb.so");


    List<Artifact> middleman2 = getAggregatingMiddleman(pyRuleB, true);
    assertEquals(
        Iterables.getOnlyElement(middleman1).getExecPathString(),
        Iterables.getOnlyElement(middleman2).getExecPathString());
  }

  /**
   * Regression test: tests that Java CPU configurations are taken into account when
   * generating a rule's aggregating middleman, so that otherwise equivalent rules can sustain
   * distinct middlemen.
   */
  @Test
  public void testJavaCcConfigurations() throws Exception {
    setupJavaPythonCcConfigurationFiles();

    // Equivalent cc / Java configurations:

    ConfiguredTarget ccRuleA = getConfiguredTarget("//foo:liba.so");
    List<Artifact> middleman1 = getAggregatingMiddleman(ccRuleA, true);
    try {
      ConfiguredTarget ccRuleB = getConfiguredTarget("//foo:libb.so");
      getAggregatingMiddleman(ccRuleB, true);
      analysisEnvironment.registerWith(getMutableActionGraph());
      fail("Expected ActionConflictException due to same middleman artifact with different files");
    } catch (UncheckedActionConflictException e) {
      // Expected failure: same "purpose" and root directory sent to the middleman generator
      // (which results in the same output artifact), but different rules / middleman inputs.
    }

    // This should succeed because the java_binary's middleman is under the Java configuration's
    // internal directory, while the cc_binary's middleman is under the cc config's directory.
    ConfiguredTarget javaRuleB = getDirectPrerequisite(
        getConfiguredTarget("//foo:d"), "//foo:libb.so");
    List<Artifact> middleman2 = getAggregatingMiddleman(javaRuleB, false);
    assertFalse(
        Iterables.getOnlyElement(middleman1)
            .getExecPathString()
            .equals(Iterables.getOnlyElement(middleman2).getExecPathString()));
  }

  private void setupJavaPythonCcConfigurationFiles() throws IOException {
    scratch.file(
        "foo/BUILD",
        "cc_binary(name = 'liba.so',",
        "    linkshared = 1,",
        "    srcs = ['a.cc'])",
        "cc_binary(name = 'libb.so',",
        "    linkshared = 1,",
        "    srcs = ['b.cc'])",
        "py_binary(name = 'c',",
        "    data = [':libb.so'],",
        "    srcs = ['c.py'])",
        "java_binary(name = 'd',",
        "    srcs = ['d.java'],",
        "    data = [':libb.so'],",
        "    main_class = 'd')");
  }
}
