// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.jacoco.core.analysis.Analyzer;
import org.jacoco.core.analysis.IClassCoverage;
import org.jacoco.core.analysis.ICoverageVisitor;
import org.jacoco.core.data.ExecutionData;
import org.jacoco.core.data.ExecutionDataStore;
import org.jacoco.core.internal.InputStreams;
import org.jacoco.core.internal.data.CRC64;
import org.jacoco.core.internal.flow.ClassProbesAdapter;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Opcodes;

/**
 * Custom analyzer to calculate the coverage of source files in the form we require.
 *
 * <p>Jacoco does not expose coverage for individual branches, reporting how many branches were
 * covered on a line (and how many there are) and not reporting which branches were covered. In
 * order to be able to merge coverage reports correctly we need to know which branches were covered
 * on each line. Since we have to process the coverage data for branches we might as well record
 * line coverage as well.
 *
 * <p>Reuse the Analyzer class from Jacoco to avoid duplicating the content detection logic.
 * Override the main {@code Analyzer.analyzeClass} method which does the main work.
 */
public class CoverageAnalyzer extends Analyzer {

  private final ExecutionDataStore executionData;
  private final Map<String, CoverageData.Builder> classCoverageData;

  public CoverageAnalyzer(final ExecutionDataStore executionData) {
    super(
        executionData,
        new ICoverageVisitor() {
          @Override
          public void visitCoverage(IClassCoverage coverage) {}
        });
    this.executionData = executionData;
    this.classCoverageData = new TreeMap<>();
  }

  // Override all analyzeClass methods.
  @Override
  public void analyzeClass(final InputStream input, final String location) throws IOException {
    final byte[] buffer;
    try {
      buffer = InputStreams.readFully(input);
    } catch (final IOException e) {
      throw analyzerError(location, e);
    }
    analyzeClass(buffer, location);
  }

  @Override
  public void analyzeClass(final byte[] buffer, final String location) throws IOException {
    try {
      analyzeClass(buffer);
    } catch (final RuntimeException cause) {
      throw analyzerError(location, cause);
    }
  }

  private void analyzeClass(final ClassReader reader) {
    final ClassProbesMapper mapper = new ClassProbesMapper(reader.getClassName());
    final ClassProbesAdapter adapter = new ClassProbesAdapter(mapper, false);

    // Skip synthetic classes and modules.
    if ((reader.getAccess() & Opcodes.ACC_SYNTHETIC) != 0
        || (reader.getAccess() & Opcodes.ACC_MODULE) != 0) {
      return;
    }

    reader.accept(adapter, 0); // Read the class using the ClassProbesMapper visitor

    final Map<Integer, BranchExpression> lineToBranchExpression = mapper.getBranchExpressions();
    final Map<Integer, CoverageExpression> lineToCoverageExpression = mapper.getLineExpressions();
    final List<MethodInfo> methods = mapper.getMethods();

    long classid = CRC64.classId(reader.b);
    ExecutionData classExecutionData = executionData.get(classid);

    // It's possible our class was never executed or that we're generating a baseline coverage
    // report but we still need to perform the analysis run.
    boolean[] probes = null;
    if (classExecutionData != null) {
      probes = classExecutionData.getProbes();
    }

    String packageName = mapper.getPackageName();
    String sourceFileName = mapper.getSourceFileName();
    if (!packageName.isEmpty()) {
      sourceFileName = packageName + "/" + sourceFileName;
    }

    CoverageData.Builder coverageBuilder = classCoverageData.get(sourceFileName);
    if (coverageBuilder == null) {
      coverageBuilder = CoverageData.builder();
    }

    for (Map.Entry<Integer, CoverageExpression> entry : lineToCoverageExpression.entrySet()) {
      int line = entry.getKey();
      CoverageExpression exp = entry.getValue();
      coverageBuilder.addLine(line, exp.eval(probes));
    }
    for (Map.Entry<Integer, BranchExpression> entry : lineToBranchExpression.entrySet()) {
      int line = entry.getKey();
      BranchExpression branchExpression = entry.getValue();
      List<CoverageExpression> branches = branchExpression.getBranches();
      for (CoverageExpression branch : branches) {
        coverageBuilder.addBranch(line, branch.eval(probes));
      }
    }
    for (MethodInfo method : methods) {
      coverageBuilder.addMethod(
          method.name(), method.startLine(), method.coverageExpression().eval(probes));
    }
    if (!coverageBuilder.isEmpty()) {
      classCoverageData.put(sourceFileName, coverageBuilder);
    }
  }

  private void analyzeClass(final byte[] source) {
    final ClassReader reader = new ClassReader(source);
    analyzeClass(reader);
  }

  private IOException analyzerError(final String location, final Exception cause) {
    final IOException ex = new IOException(String.format("Error while analyzing %s.", location));
    ex.initCause(cause);
    return ex;
  }

  /** Returns the coverage data for each source file. */
  public ImmutableMap<String, CoverageData> getCoverage() {
    return ImmutableMap.copyOf(
        Maps.transformValues(classCoverageData, CoverageData.Builder::build));
  }
}
