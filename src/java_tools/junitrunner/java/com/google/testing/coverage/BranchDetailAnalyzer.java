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

/**
 * Analyzer that process the branch coverage detail information.
 *
 * <p>Reuse the Analyzer class from Jacoco to avoid duplicating the content detection logic.
 * Override the main {@code Analyzer.analyzeClass} method which does the main work.
 */
public class BranchDetailAnalyzer extends Analyzer {

  private final ExecutionDataStore executionData;
  private final Map<String, BranchCoverageDetail> branchDetails;

  public BranchDetailAnalyzer(final ExecutionDataStore executionData) {
    super(
        executionData,
        new ICoverageVisitor() {
          @Override
          public void visitCoverage(IClassCoverage coverage) {}
        });
    this.executionData = executionData;
    this.branchDetails = new TreeMap<String, BranchCoverageDetail>();
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

  @Override
  public void analyzeClass(final ClassReader reader) {
    final Map<Integer, BranchExp> lineToBranchExp = mapProbes(reader);

    long classid = CRC64.classId(reader.b);
    ExecutionData classData = executionData.get(classid);
    if (classData == null) {
      return;
    }
    boolean[] probes = classData.getProbes();

    BranchCoverageDetail detail = new BranchCoverageDetail();

    for (Map.Entry<Integer, BranchExp> entry : lineToBranchExp.entrySet()) {
      int line = entry.getKey();
      BranchExp branchExp = entry.getValue();
      List<CovExp> branches = branchExp.getBranches();

      detail.setBranches(line, branches.size());
      for (int branchIdx = 0; branchIdx < branches.size(); branchIdx++) {
        if (branches.get(branchIdx).eval(probes)) {
          detail.setTakenBit(line, branchIdx);
        }
      }
    }
    if (detail.linesWithBranches().size() > 0) {
      branchDetails.put(reader.getClassName(), detail);
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

  // Generate the line to probeExp map so that we can evaluate the coverage.
  private Map<Integer, BranchExp> mapProbes(final ClassReader reader) {
    final ClassProbesMapper mapper = new ClassProbesMapper();
    final ClassProbesAdapter adapter = new ClassProbesAdapter(mapper, false);
    reader.accept(adapter, 0);

    return mapper.result();
  }

  public Map<String, BranchCoverageDetail> getBranchDetails() {
    return branchDetails;
  }
}
