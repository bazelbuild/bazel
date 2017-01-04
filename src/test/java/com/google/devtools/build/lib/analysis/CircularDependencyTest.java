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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that check that dependency cycles are reported correctly.
 */
@RunWith(JUnit4.class)
public class CircularDependencyTest extends BuildViewTestCase {

  @Test
  public void testOneRuleCycle() throws Exception {
    checkError(
        "cycle",
        "foo.g",
        //error message
        selfEdgeMsg("//cycle:foo.g"),
        // Rule
        "genrule(name = 'foo.g',",
        "        outs = ['Foo.java'],",
        "        srcs = ['foo.g'],",
        "        cmd = 'cat $(SRCS) > $<' )");
  }

  @Test
  public void testDirectPackageGroupCycle() throws Exception {
    checkError(
        "cycle",
        "melon",
        selfEdgeMsg("//cycle:moebius"),
        "package_group(name='moebius', packages=[], includes=['//cycle:moebius'])",
        "sh_library(name='melon', visibility=[':moebius'])");
  }

  @Test
  public void testThreeLongPackageGroupCycle() throws Exception {
    String expectedEvent =
        "cycle in dependency graph:\n"
            + "    //cycle:superman\n"
            + ".-> //cycle:rock\n"
            + "|   //cycle:paper\n"
            + "|   //cycle:scissors\n"
            + "`-- //cycle:rock";
    checkError(
        "cycle",
        "superman",
        expectedEvent,
        "# dummy line",
        "package_group(name='paper', includes=['//cycle:scissors'])",
        "package_group(name='rock', includes=['//cycle:paper'])",
        "package_group(name='scissors', includes=['//cycle:rock'])",
        "sh_library(name='superman', visibility=[':rock'])");

    Event foundEvent = null;
    for (Event event : eventCollector) {
      if (event.getMessage().contains(expectedEvent)) {
        foundEvent = event;
        break;
      }
    }

    assertNotNull(foundEvent);
    Location location = foundEvent.getLocation();
    assertEquals(3, location.getStartLineAndColumn().getLine());
    assertEquals("/workspace/cycle/BUILD", location.getPath().toString());
  }

  /**
   * Test to detect implicit input/output file overlap in rules.
   */
  @Test
  public void testOneRuleImplicitCycleJava() throws Exception {
    Package pkg =
        createScratchPackageForImplicitCycle(
            "cycle", "java_library(name='jcyc',", "      srcs = ['libjcyc.jar', 'foo.java'])");
    try {
      pkg.getTarget("jcyc");
      fail();
    } catch (NoSuchTargetException e) {
      /* ok */
    }
    assertTrue(pkg.containsErrors());
    assertContainsEvent("rule 'jcyc' has file 'libjcyc.jar' as both an" + " input and an output");
  }

  /**
   * Test not to detect implicit input/output file overlap in rules,
   * when coming from a different package.
   */
  @Test
  public void testInputOutputConflictDifferentPackage() throws Exception {
    Package pkg =
        createScratchPackageForImplicitCycle(
            "googledata/xxx",
            "genrule(name='geo',",
            "    srcs = ['//googledata/geo:geo_info.txt'],",
            "    outs = ['geoinfo.txt'],",
            "    cmd = '$(SRCS) > $@')");
    assertFalse(pkg.containsErrors());
  }

  @Test
  public void testTwoRuleCycle() throws Exception {
    scratchRule("b", "rule2", "cc_library(name='rule2',", "           deps=['//a:rule1'])");

    checkError(
        "a",
        "rule1",
        "in cc_library rule //a:rule1: cycle in dependency graph:\n"
            + ".-> //a:rule1\n"
            + "|   //b:rule2\n"
            + "`-- //a:rule1",
        "cc_library(name='rule1',",
        "           deps=['//b:rule2'])");
  }

  @Test
  public void testTwoRuleCycle2() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD", "java_library(name='x', deps=['y'])", "java_library(name='y', deps=['x'])");
    getConfiguredTarget("//x");
    assertContainsEvent("in java_library rule //x:x: cycle in dependency graph");
  }

  @Test
  public void testIndirectOneRuleCycle() throws Exception {
    scratchRule(
        "cycle",
        "foo.h",
        "genrule(name = 'foo.h',",
        "      outs = ['bar.h'],",
        "      srcs = ['foo.h'],",
        "      cmd = 'cp $< $@')");
    checkError(
        "main",
        "mygenrule",
        //error message
        selfEdgeMsg("//cycle:foo.h"),
        // Rule
        "genrule(name='mygenrule',",
        "      outs = ['baz.h'],",
        "      srcs = ['//cycle:foo.h'],",
        "      cmd = 'cp $< $@')");
  }

  private String selfEdgeMsg(String label) {
    return label + " [self-edge]";
  }

  // Regression test for: "IllegalStateException in
  // AbstractConfiguredTarget.initialize()".
  // Failure to mark all cycle-forming nodes when there are *two* cycles led to
  // an attempt to initialise a node we'd already visited.
  @Test
  public void testTwoCycles() throws Exception {
    reporter.removeHandler(failFastHandler); // expect errors
    scratch.file(
        "x/BUILD",
        "genrule(name='b', srcs=['c'], tools=['c'], outs=['b.out'], cmd=':')",
        "genrule(name='c', srcs=['b.out'], outs=[], cmd=':')");
    getConfiguredTarget("//x:b"); // doesn't crash!
    assertContainsEvent("cycle in dependency graph");
  }

  @Test
  public void testAspectCycle() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("x/BUILD",
        "load('//x:x.bzl', 'aspected', 'plain')",
        // Using data= makes the dependency graph clearer because then the aspect does not propagate
        // from aspectdep through a to b (and c)
        "plain(name = 'a', noaspect_deps = [':b'])",
        "aspected(name = 'b', aspect_deps = ['c'])",
        "plain(name = 'c')",
        "plain(name = 'aspectdep', aspect_deps = ['a'])");

    scratch.file("x/x.bzl",
        "def _impl(ctx):",
        "    return struct()",
        "",
        "rule_aspect = aspect(",
        "    implementation = _impl,",
        "    attr_aspects = ['aspect_deps'],",
        "    attrs = { '_implicit': attr.label(default = Label('//x:aspectdep')) })",
        "",
        "plain = rule(",
        "    implementation = _impl,",
        "    attrs = { 'aspect_deps': attr.label_list(), 'noaspect_deps': attr.label_list() })",
        "",
        "aspected = rule(",
        "    implementation = _impl,",
        "    attrs = { 'aspect_deps': attr.label_list(aspects = [rule_aspect]) })");

    getConfiguredTarget("//x:a");
    assertContainsEvent("cycle in dependency graph");
    assertContainsEvent("//x:c with aspect //x:x.bzl%rule_aspect");
  }
}
