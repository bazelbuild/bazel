// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for MC/DC coverage.
 *
 * <p>Tests simulate coverage scenarios for a decision function:
 * {@code bool decision(bool a, bool b, bool c) { return (a && b) || c; }}
 *
 * <p>MC/DC (Modified Condition/Decision Coverage) ensures each condition independently affects the
 * decision outcome. For the expression {@code (a && b) || c}, MC/DC coverage requires test pairs
 * where flipping each condition changes the result:
 *
 * <ul>
 *   <li><b>Condition 'a' changes outcome:</b> T: (T,T,F)→T vs F: (F,T,F)→F [a flips, b=T, c=F]
 *   <li><b>Condition 'b' changes outcome:</b> T: (T,T,F)→T vs F: (T,F,F)→F [b flips, a=T, c=F]
 *   <li><b>Condition 'c' changes outcome:</b> T: (F,F,T)→T vs F: (F,F,F)→F [c flips, a=F, b=F]
 * </ul>
 *
 * <p>The test {@link #testFullMcdcCoverage()} demonstrates full MC/DC coverage achieved by these
 * 5 test cases: decision(T,T,F), decision(F,T,F), decision(T,F,F), decision(F,F,T),
 * decision(F,F,F). This achieves 100% MC/DC coverage with each condition demonstrating independent
 * effect on the outcome.
 *
 * <p>LLVM's MC/DC implementation tracks this as 6 records (3 conditions × 2 senses), where each
 * record represents a condition evaluated to true or false in a way that independently affects the
 * decision outcome.
 */
@RunWith(JUnit4.class)
public class McdcIntegrationTest {

  @Test
  public void testFullMcdcCoverage() throws IOException {
    ImmutableList<String> inputTracefile =
        ImmutableList.of(
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:5,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,1,0,'a' in '(a && b) || c'",  // condition a: true path hit once
            "MCDC:5,3,f,1,0,'a' in '(a && b) || c'",  // condition a: false path hit once
            "MCDC:5,3,t,1,1,'b' in '(a && b) || c'",  // condition b: true path hit once
            "MCDC:5,3,f,1,1,'b' in '(a && b) || c'",  // condition b: false path hit once
            "MCDC:5,3,t,1,2,'c' in '(a && b) || c'",  // condition c: true path hit once
            "MCDC:5,3,f,1,2,'c' in '(a && b) || c'",  // condition c: false hit once
            "MCF:6",  // 6 MC/DC records found (3 conditions × 2 senses)
            "MCH:6",  // All 6 records hit
            "DA:5,5",
            "LF:1",
            "LH:1",
            "end_of_record");

    // Parse and verify coverage data
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(inputTracefile).getBytes(UTF_8)));
    assertThat(sourceFiles).hasSize(1);
    SourceFileCoverage sourceFile = sourceFiles.get(0);
    assertThat(sourceFile.sourceFileName()).isEqualTo("decision_logic.c");
    assertThat(sourceFile.nrMcdcFound()).isEqualTo(6);
    assertThat(sourceFile.nrMcdcHit()).isEqualTo(6);

    // Verify parsed MC/DC records
    List<McdcCoverage> mcdcRecords = sourceFile.getAllMcdc().stream().collect(Collectors.toList());
    assertThat(mcdcRecords).hasSize(6);
    assertThat(mcdcRecords)
        .containsAtLeast(
            McdcCoverage.create(5, 3, 't', 1, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 1, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 1, 2, "'c' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 2, "'c' in '(a && b) || c'"));
    assertThat(mcdcRecords).hasSize(6);

    // Verify round-trip printing
    Coverage coverage = new Coverage();
    coverage.add(sourceFile);
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    LcovPrinter.print(outputStream, coverage);
    Iterable<String> outputLines = Splitter.on('\n').split(outputStream.toString(UTF_8).strip());
    assertThat(outputLines)
        .containsAtLeast(
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:5,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,1,0,'a' in '(a && b) || c'",
            "MCDC:5,3,f,1,0,'a' in '(a && b) || c'",
            "MCDC:5,3,t,1,1,'b' in '(a && b) || c'",
            "MCDC:5,3,f,1,1,'b' in '(a && b) || c'",
            "MCDC:5,3,t,1,2,'c' in '(a && b) || c'",
            "MCDC:5,3,f,1,2,'c' in '(a && b) || c'",
            "MCF:6",
            "MCH:6",
            "DA:5,5",
            "LH:1",
            "LF:1",
            "end_of_record");
  }

  @Test
  public void testMcdcFullCoverageAfterMerging() throws IOException {
    // Test suite 1 covers conditions c and b (false sense).
    ImmutableList<String> tracefile1 =
        ImmutableList.of(
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:2,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,0,0,'a' in '(a && b) || c'",  // condition a: true not hit
            "MCDC:5,3,f,0,0,'a' in '(a && b) || c'",  // condition a: false not hit
            "MCDC:5,3,t,0,1,'b' in '(a && b) || c'",  // condition b: true not hit
            "MCDC:5,3,f,1,1,'b' in '(a && b) || c'",  // condition b: false hit once
            "MCDC:5,3,t,1,2,'c' in '(a && b) || c'",  // condition c: true hit once
            "MCDC:5,3,f,1,2,'c' in '(a && b) || c'",  // condition c: false hit once
            "MCF:6",
            "MCH:4",  // Partial coverage: 4 out of 6 records hit
            "DA:5,2",
            "LF:1",
            "LH:1",
            "end_of_record");

    // Test suite 2 covers conditions a and b (true sense).
    ImmutableList<String> tracefile2 =
        ImmutableList.of(
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:2,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,1,0,'a' in '(a && b) || c'",  // condition a: true hit once
            "MCDC:5,3,f,1,0,'a' in '(a && b) || c'",  // condition a: false hit once
            "MCDC:5,3,t,1,1,'b' in '(a && b) || c'",  // condition b: true hit once
            "MCDC:5,3,f,0,1,'b' in '(a && b) || c'",  // condition b: false not hit
            "MCDC:5,3,t,0,2,'c' in '(a && b) || c'",  // condition c: true not hit
            "MCDC:5,3,f,1,2,'c' in '(a && b) || c'",  // condition c: false hit once
            "MCF:6",
            "MCH:4",  // Partial coverage: 4 out of 6 records hit
            "DA:5,2",
            "LF:1",
            "LH:1",
            "end_of_record");

    // Parse and merge coverage data
    List<SourceFileCoverage> sources1 =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(tracefile1).getBytes(UTF_8)));
    List<SourceFileCoverage> sources2 =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(tracefile2).getBytes(UTF_8)));
    SourceFileCoverage merged = SourceFileCoverage.merge(sources1.get(0), sources2.get(0));

    assertThat(merged.nrMcdcFound()).isEqualTo(6);
    assertThat(merged.nrMcdcHit()).isEqualTo(6);

    List<McdcCoverage> mcdcRecords = merged.getAllMcdc().stream().collect(Collectors.toList());
    assertThat(mcdcRecords)
        .containsAtLeast(
            McdcCoverage.create(5, 3, 't', 1, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 1, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 1, 2, "'c' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 2, 2, "'c' in '(a && b) || c'"));
    assertThat(mcdcRecords).hasSize(6);
  }

  @Test
  public void testRoundtripMcdcDataSurvivesParsingAndPrinting() throws IOException {
    ImmutableList<String> tracefile =
        ImmutableList.of(
            "TN:",
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:8,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,3,0,'a' in '(a && b) || c'",  // condition a: true hit 3 times
            "MCDC:5,3,f,2,0,'a' in '(a && b) || c'",  // condition a: false hit 2 times
            "MCDC:5,3,t,4,1,'b' in '(a && b) || c'",  // condition b: true hit 4 times
            "MCDC:5,3,f,1,1,'b' in '(a && b) || c'",  // condition b: false hit once
            "MCDC:5,3,t,2,2,'c' in '(a && b) || c'",  // condition c: true hit 2 times
            "MCDC:5,3,f,3,2,'c' in '(a && b) || c'",  // condition c: false hit 3 times
            "MCF:6",  // 6 MC/DC records (3 conditions × 2 senses)
            "MCH:6",  // All 6 records hit
            "DA:5,8",
            "LF:1",
            "LH:1",
            "end_of_record");

    // Parse and verify coverage data
    List<SourceFileCoverage> sourceFiles =
        LcovParser.parse(new ByteArrayInputStream(Joiner.on("\n").join(tracefile).getBytes(UTF_8)));
    SourceFileCoverage sourceFile = sourceFiles.get(0);
    assertThat(sourceFile.sourceFileName()).isEqualTo("decision_logic.c");
    assertThat(sourceFile.nrMcdcFound()).isEqualTo(6);
    assertThat(sourceFile.nrMcdcHit()).isEqualTo(6);

    List<McdcCoverage> mcdcRecords = sourceFile.getAllMcdc().stream().collect(Collectors.toList());
    assertThat(mcdcRecords)
        .containsAtLeast(
            McdcCoverage.create(5, 3, 't', 3, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 2, 0, "'a' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 4, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 1, 1, "'b' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 't', 2, 2, "'c' in '(a && b) || c'"),
            McdcCoverage.create(5, 3, 'f', 3, 2, "'c' in '(a && b) || c'"));
    assertThat(mcdcRecords).hasSize(6);

    // Verify round-trip printing
    Coverage coverage = new Coverage();
    coverage.add(sourceFile);
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    LcovPrinter.print(outputStream, coverage);
    String output = outputStream.toString(UTF_8);

    // Verify output format
    Iterable<String> outputLines = Splitter.on('\n').split(output.strip());
    assertThat(outputLines)
        .containsAtLeast(
            "SF:decision_logic.c",
            "FN:5,decision",
            "FNDA:8,decision",
            "FNF:1",
            "FNH:1",
            "MCDC:5,3,t,3,0,'a' in '(a && b) || c'",
            "MCDC:5,3,f,2,0,'a' in '(a && b) || c'",
            "MCDC:5,3,t,4,1,'b' in '(a && b) || c'",
            "MCDC:5,3,f,1,1,'b' in '(a && b) || c'",
            "MCDC:5,3,t,2,2,'c' in '(a && b) || c'",
            "MCDC:5,3,f,3,2,'c' in '(a && b) || c'",
            "MCF:6",
            "MCH:6",
            "DA:5,8",
            "LH:1",
            "LF:1",
            "end_of_record");

    // Verify round-trip parsing
    List<SourceFileCoverage> reparsed =
        LcovParser.parse(new ByteArrayInputStream(output.getBytes(UTF_8)));
    assertThat(reparsed).hasSize(1);
    SourceFileCoverage reparsedFile = reparsed.get(0);
    assertThat(reparsedFile.nrMcdcFound()).isEqualTo(sourceFile.nrMcdcFound());
    assertThat(reparsedFile.nrMcdcHit()).isEqualTo(sourceFile.nrMcdcHit());
    assertThat(reparsedFile.getAllMcdc().stream().collect(Collectors.toList()))
        .containsExactlyElementsIn(mcdcRecords);
  }
}
