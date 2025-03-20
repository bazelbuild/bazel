// Copyright 2020 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import org.jacoco.core.analysis.IBundleCoverage;
import org.jacoco.core.analysis.IClassCoverage;
import org.jacoco.core.analysis.IPackageCoverage;
import org.jacoco.report.IReportVisitor;
import org.jacoco.report.ISourceFileLocator;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the uninstrumented class processing logic in {@link JacocoLCOVFormatter}. */
@RunWith(JUnit4.class)
public class JacocoLCOVFormatterUninstrumentedTest {

  private StringWriter writer;
  private IBundleCoverage mockBundle;

  private static IClassCoverage mockIClassCoverage(
      String className, String packageName, String sourceFileName) {
    IClassCoverage mocked = mock(IClassCoverage.class);
    when(mocked.getName()).thenReturn(className);
    when(mocked.getPackageName()).thenReturn(packageName);
    when(mocked.getSourceFileName()).thenReturn(sourceFileName);
    return mocked;
  }

  private Description createSuiteDescription(String name) {
    Description suite = Description.createSuiteDescription(name);
    suite.addChild(Description.createTestDescription(Object.class, "child"));
    return suite;
  }

  @Before
  public void setupTest() {
    // Initialize writer for storing coverage report outputs
    writer = new StringWriter();
    // Initialize mock Jacoco bundle containing the mock coverage
    // Classes
    List<IClassCoverage> mockClassCoverages =
        Arrays.asList(mockIClassCoverage("Foo", "com/example", "Foo.java"));
    // Package
    IPackageCoverage mockPackageCoverage = mock(IPackageCoverage.class);
    when(mockPackageCoverage.getClasses()).thenReturn(mockClassCoverages);
    // Bundle
    mockBundle = mock(IBundleCoverage.class);
    when(mockBundle.getPackages()).thenReturn(Arrays.asList(mockPackageCoverage));
  }

  @Test
  public void testVisitBundleWithSimpleUnixPath() throws IOException {
    // Paths
    ImmutableSet<String> execPaths = ImmutableSet.of("/parent/dir/com/example/Foo.java");
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    for (String sourcePath : execPaths) {
      assertThat(coverageOutput).contains(sourcePath);
    }
  }

  @Test
  public void testVisitBundleWithSimpleWindowsPath() throws IOException {
    // Paths
    ImmutableSet<String> execPaths = ImmutableSet.of("C:/parent/dir/com/example/Foo.java");
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    for (String sourcePath : execPaths) {
      assertThat(coverageOutput).contains(sourcePath);
    }
  }

  @Test
  public void testVisitBundleWithMappedUnixPath() throws IOException {
    // Paths
    String srcPath = "/some/other/dir/Foo.java";
    ImmutableSet<String> execPaths = ImmutableSet.of(srcPath + "////com/example/Foo.java");
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).contains(srcPath);
  }

  @Test
  public void testVisitBundleWithMappedWindowsPath() throws IOException {
    // Paths
    String srcPath = "C:/some/other/dir/Foo.java";
    ImmutableSet<String> execPaths = ImmutableSet.of(srcPath + "////com/example/Foo.java");
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).contains(srcPath);
  }

  @Test
  public void testVisitBundleWithNoMatchHasEmptyOutput() throws IOException {
    // Non-matching path
    ImmutableSet<String> execPaths = ImmutableSet.of("/path/does/not/match/anything.txt");
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).isEmpty();
  }

  @Test
  public void testVisitBundleWithNoExecPathsHasEmptyOutput() throws IOException {
    // Empty list of exec paths
    ImmutableSet<String> execPaths = ImmutableSet.of();
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).isEmpty();
  }

  @Test
  public void testVisitBundleWithoutExecPathsDoesNotPruneOutput() throws IOException {
    // No paths, don't attempt to demangle paths and prune the output, just output with
    // class-paths as is.
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter();
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).isNotEmpty();
  }

  @Test
  public void testVisitBundleWithExactMatch() throws IOException {
    // It's possible, albeit unlikely, that the execPath and the package based path match exactly
    String srcPath = "com/example/Foo.java";
    ImmutableSet<String> execPaths = ImmutableSet.of(srcPath);
    JacocoLCOVFormatter formatter = new JacocoLCOVFormatter(execPaths);
    IReportVisitor visitor =
        formatter.createVisitor(
            new PrintWriter(writer), new TreeMap<String, BranchCoverageDetail>());

    visitor.visitBundle(mockBundle, mock(ISourceFileLocator.class));
    visitor.visitEnd();

    String coverageOutput = writer.toString();
    assertThat(coverageOutput).contains(srcPath);
  }
}
