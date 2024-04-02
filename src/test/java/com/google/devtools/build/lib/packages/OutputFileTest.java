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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.util.TargetDataSubject.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class OutputFileTest extends PackageLoadingTestCase {
  private Package pkg;
  private Rule rule;

  @Before
  public final void createRule() throws Exception {
    scratch.file("pkg/BUILD", "genrule(name='foo', srcs=[], cmd='', outs=['x', 'subdir/y'])");
    this.rule = (Rule) getTarget("//pkg:foo");
    this.pkg = rule.getPackage();
    assertNoEvents();
  }

  private void checkTargetRetainsGeneratingRule(OutputFile output) throws Exception {
    assertThat(output.getGeneratingRule()).isSameInstanceAs(rule);
  }

  private void checkName(OutputFile output, String expectedName) throws Exception {
    assertThat(output.getName()).isEqualTo(expectedName);
  }

  private void checkLabel(OutputFile output, String expectedLabelString) throws Exception {
    assertThat(output.getLabel().toString()).isEqualTo(expectedLabelString);
  }

  @Test
  public void testGetAssociatedRule() throws Exception {
    assertThat(pkg.getTarget("x").getAssociatedRule()).isSameInstanceAs(rule);
  }

  @Test
  public void testOutputFileInPackageDir() throws Exception {
    OutputFile outputFileX = (OutputFile) pkg.getTarget("x");
    checkTargetRetainsGeneratingRule(outputFileX);
    checkName(outputFileX, "x");
    checkLabel(outputFileX, "//pkg:x");
    assertThat(outputFileX.getTargetKind()).isEqualTo("generated file");
  }

  @Test
  public void testOutputFileInSubdirectory() throws Exception {
    OutputFile outputFileY = (OutputFile) pkg.getTarget("subdir/y");
    checkTargetRetainsGeneratingRule(outputFileY);
    checkName(outputFileY, "subdir/y");
    checkLabel(outputFileY, "//pkg:subdir/y");
  }

  @Test
  public void testEquivalenceRelation() throws Exception {
    OutputFile outputFileX1 = (OutputFile) pkg.getTarget("x");
    OutputFile outputFileX2 = (OutputFile) pkg.getTarget("x");
    OutputFile outputFileY1 = (OutputFile) pkg.getTarget("subdir/y");
    OutputFile outputFileY2 = (OutputFile) pkg.getTarget("subdir/y");
    assertThat(outputFileX2).isSameInstanceAs(outputFileX1);
    assertThat(outputFileY2).isSameInstanceAs(outputFileY1);
    new EqualsTester()
        .addEqualityGroup(outputFileX1, outputFileX2)
        .addEqualityGroup(outputFileY1, outputFileY2)
        .testEquals();
  }

  @Test
  public void testDuplicateOutputFilesInDifferentRules() throws Exception {
    scratch.file(
        "two_outs/BUILD",
        """
        genrule(
            name = "a",
            outs = ["out"],
            cmd = "ls >$(location out)",
        )

        genrule(
            name = "b",
            outs = ["out"],
            cmd = "ls >$(location out)",
        )
        """);

    reporter.removeHandler(failFastHandler);
    getTarget("//two_outs:BUILD");
    assertContainsEvent(
        "generated file 'out' in rule 'b' conflicts with "
            + "existing generated file from rule 'a'");
  }

  @Test
  public void testOutputFileNameConflictsWithExistingRule() throws Exception {
    scratch.file(
        "out_is_rule/BUILD",
        """
        genrule(
            name = "a",
            outs = ["out"],
            cmd = "ls >$(location out)",
        )

        genrule(
            name = "b",
            outs = ["a"],
            cmd = "ls >$(location out)",
        )
        """);

    reporter.removeHandler(failFastHandler);
    getTarget("//out_is_rule:BUILD");
    assertContainsEvent("generated file 'a' in rule 'b' conflicts with existing genrule rule");
  }

  @Test
  public void testDuplicateOutputFilesInSameRule() throws Exception {
    scratch.file(
        "two_outs/BUILD",
        """
        genrule(
            name = "a",
            outs = [
                "out",
                "out",
            ],
            cmd = "ls >$(location out)",
        )
        """);
    reporter.removeHandler(failFastHandler);
    getTarget("//two_outs:BUILD");
    assertContainsEvent("rule 'a' has more than one generated file named 'out'");
  }

  @Test
  public void testOutputFileWithIllegalName() throws Exception {
    scratch.file("bad_out_name/BUILD", "genrule(name='a', cmd='ls', outs=['!@#:'])");
    reporter.removeHandler(failFastHandler);
    getTarget("//bad_out_name:BUILD");
    assertContainsEvent("invalid label '!@#:'");
  }

  @Test
  public void testOutputFileWithCrossPackageLabel() throws Exception {
    scratch.file("cross_package_out/BUILD", "genrule(name='a', cmd='ls', outs=['//foo:bar'])");
    reporter.removeHandler(failFastHandler);
    getTarget("//cross_package_out:BUILD");
    assertContainsEvent("label '//foo:bar' is not in the current package");
  }

  @Test
  public void testOutputFileNamedBUILD() throws Exception {
    scratch.file("output_called_build/BUILD", "genrule(name='a', cmd='ls', outs=['BUILD'])");
    reporter.removeHandler(failFastHandler);
    getTarget("//output_called_build:BUILD");
    assertContainsEvent("generated file 'BUILD' in rule 'a' conflicts with existing source file");
  }

  @Test
  public void testReduceForSerialization() throws Exception {
    var outputFileX = pkg.getTarget("x");
    assertThat(outputFileX).hasSamePropertiesAs(outputFileX.reduceForSerialization());
    var outputFileY = pkg.getTarget("subdir/y");
    assertThat(outputFileY).hasSamePropertiesAs(outputFileY.reduceForSerialization());
  }
}
