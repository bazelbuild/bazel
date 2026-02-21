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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link McdcCoverageData}. */
@RunWith(JUnit4.class)
public class McdcCoverageDataTest {

  @Test
  public void testCreate_initialStateIsEmpty() {
    McdcCoverageData data = McdcCoverageData.create();

    assertThat(data.size()).isEqualTo(0);
    assertThat(data.nrMcdcHit()).isEqualTo(0);
    assertThat(data.getAllRecords()).isEmpty();
  }

  @Test
  public void testAddMcdc_singleRecord() {
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");

    assertThat(data.size()).isEqualTo(1);
    assertThat(data.nrMcdcHit()).isEqualTo(1);
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records).hasSize(1);
    assertThat(records.get(0).lineNumber()).isEqualTo(10);
    assertThat(records.get(0).groupSize()).isEqualTo(2);
    assertThat(records.get(0).sense()).isEqualTo('t');
    assertThat(records.get(0).taken()).isEqualTo(5);
    assertThat(records.get(0).index()).isEqualTo(0);
    assertThat(records.get(0).expression()).isEqualTo("a && b");
  }

  @Test
  public void testAddMcdc_multipleRecordsWithDifferentKeys() {
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 'f', 3, 1, "a && b");
    data.addMcdc(20, 3, 't', 7, 0, "c || d");

    assertThat(data.size()).isEqualTo(3);
    assertThat(data.nrMcdcHit()).isEqualTo(3);
  }

  @Test
  public void testAddMcdc_duplicateKeyMergesTakenCounts() {
    // This tests the key insight: same key = automatic merging of taken counts
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 't', 3, 0, "a && b"); // Same key, should merge

    assertThat(data.size()).isEqualTo(1); // Still only 1 record
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records).hasSize(1);
    assertThat(records.get(0).taken()).isEqualTo(8); // 5 + 3 = 8
  }

  @Test
  public void testAddMcdc_zeroTakenCountNotHit() {
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 0, 0, "a && b");
    data.addMcdc(10, 2, 'f', 5, 1, "a && b");

    assertThat(data.size()).isEqualTo(2);
    assertThat(data.nrMcdcHit()).isEqualTo(1); // Only the second one was hit
  }

  @Test
  public void testAddMcdc_emptyExpression() {
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 0, 't', 1, 0, "");

    assertThat(data.size()).isEqualTo(1);
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).expression()).isEmpty();
  }

  @Test
  public void testAddMcdc_nullExpressionThrowsException() {
    McdcCoverageData data = McdcCoverageData.create();

    assertThrows(
        IllegalArgumentException.class, () -> data.addMcdc(10, 2, 't', 5, 0, null));
  }

  @Test
  public void testAddMcdc_complexExpression() {
    McdcCoverageData data = McdcCoverageData.create();
    String complexExpr = "((a || b) && (c || d)) || e";

    data.addMcdc(50, 5, 'f', 42, 3, complexExpr);

    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).expression()).isEqualTo(complexExpr);
  }

  @Test
  public void testAddMcdc_viaMcdcCoverageObject() {
    McdcCoverageData data = McdcCoverageData.create();
    McdcCoverage record = McdcCoverage.create(10, 2, 't', 5, 0, "a && b");

    data.addMcdc(record);

    assertThat(data.size()).isEqualTo(1);
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).lineNumber()).isEqualTo(10);
    assertThat(records.get(0).taken()).isEqualTo(5);
  }

  @Test
  public void testAddMcdc_differentSenseCharactersAreDifferentKeys() {
    // Tests that sense 't' and 'f' are treated as different keys
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 'f', 3, 0, "a && b"); // Same except sense

    assertThat(data.size()).isEqualTo(2); // Different records
  }

  @Test
  public void testAddMcdc_differentIndicesAreDifferentKeys() {
    // Tests that different indices create different keys
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 't', 3, 1, "a && b"); // Same except index

    assertThat(data.size()).isEqualTo(2); // Different records
  }

  @Test
  public void testAddMcdc_differentExpressionsAreDifferentKeys() {
    // Tests that different expressions create different keys
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 't', 3, 0, "c || d"); // Same except expression

    assertThat(data.size()).isEqualTo(2); // Different records
  }

  @Test
  public void testAdd_mergesDataFromAnotherInstance() {
    McdcCoverageData data1 = McdcCoverageData.create();
    McdcCoverageData data2 = McdcCoverageData.create();

    data1.addMcdc(10, 2, 't', 5, 0, "a && b");
    data1.addMcdc(20, 3, 'f', 2, 1, "c || d");

    data2.addMcdc(10, 2, 't', 3, 0, "a && b"); // Will merge with data1
    data2.addMcdc(30, 2, 't', 1, 0, "e && f"); // New record

    data1.add(data2);

    assertThat(data1.size()).isEqualTo(3); // 2 + 1 new record
    List<McdcCoverage> records = data1.getAllRecords();
    
    // Find the merged record
    McdcCoverage merged = records.stream()
        .filter(r -> r.lineNumber() == 10 && r.index() == 0)
        .findFirst()
        .orElseThrow();
    assertThat(merged.taken()).isEqualTo(8); // 5 + 3
  }

  @Test
  public void testMerge_createsNewInstanceWithMergedData() {
    McdcCoverageData data1 = McdcCoverageData.create();
    McdcCoverageData data2 = McdcCoverageData.create();

    data1.addMcdc(10, 2, 't', 5, 0, "a && b");
    data2.addMcdc(10, 2, 't', 3, 0, "a && b");
    data2.addMcdc(20, 3, 'f', 2, 0, "c || d");

    McdcCoverageData merged = McdcCoverageData.merge(data1, data2);

    // Original instances unchanged
    assertThat(data1.size()).isEqualTo(1);
    assertThat(data2.size()).isEqualTo(2);

    // Merged instance has combined data
    assertThat(merged.size()).isEqualTo(2);
    List<McdcCoverage> records = merged.getAllRecords();
    McdcCoverage mergedRecord = records.stream()
        .filter(r -> r.lineNumber() == 10)
        .findFirst()
        .orElseThrow();
    assertThat(mergedRecord.taken()).isEqualTo(8); // 5 + 3
  }

  @Test
  public void testCopy_createsIndependentInstance() {
    McdcCoverageData original = McdcCoverageData.create();
    original.addMcdc(10, 2, 't', 5, 0, "a && b");

    McdcCoverageData copy = McdcCoverageData.copy(original);

    // Modify the copy
    copy.addMcdc(20, 3, 'f', 3, 0, "c || d");

    // Original unchanged
    assertThat(original.size()).isEqualTo(1);
    assertThat(copy.size()).isEqualTo(2);
  }

  @Test
  public void testIterator_iteratesOverAllRecords() {
    McdcCoverageData data = McdcCoverageData.create();
    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(20, 3, 'f', 3, 1, "c || d");
    data.addMcdc(30, 1, 't', 1, 0, "e");

    List<McdcCoverage> collected = new ArrayList<>();
    for (McdcCoverage record : data) {
      collected.add(record);
    }

    assertThat(collected).hasSize(3);
  }

  @Test
  public void testIterator_emptyDataSet() {
    McdcCoverageData data = McdcCoverageData.create();

    List<McdcCoverage> collected = new ArrayList<>();
    for (McdcCoverage record : data) {
      collected.add(record);
    }

    assertThat(collected).isEmpty();
  }

  @Test
  public void testCapacityExpansion_handlesGrowthCorrectly() {
    // Tests that the hash map expands when load factor exceeds 75%
    // Initial capacity is 16, so after 12 elements it should expand
    McdcCoverageData data = McdcCoverageData.create();

    // Add more than initial capacity to trigger expansion
    for (int i = 0; i < 20; i++) {
      data.addMcdc(i, 2, 't', i + 1, 0, "expr" + i);
    }

    assertThat(data.size()).isEqualTo(20);
    assertThat(data.nrMcdcHit()).isEqualTo(20);
    
    // Verify all records are accessible
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records).hasSize(20);
  }

  @Test
  public void testHashCollisions_linearProbingWorks() {
    // This test verifies that linear probing handles collisions correctly
    // We can't force collisions easily, but we can verify that many records
    // with similar keys all get stored correctly
    McdcCoverageData data = McdcCoverageData.create();

    // Add records with the same line number but different other fields
    for (int i = 0; i < 10; i++) {
      data.addMcdc(100, 2, i % 2 == 0 ? 't' : 'f', i, i, "expr" + i);
    }

    assertThat(data.size()).isEqualTo(10);
    
    // All records should be retrievable
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records).hasSize(10);
    
    // Verify each has unique combination
    long uniqueCombinations = records.stream()
        .map(r -> r.sense() + "-" + r.index() + "-" + r.expression())
        .distinct()
        .count();
    assertThat(uniqueCombinations).isEqualTo(10);
  }

  @Test
  public void testGetAllRecords_lazyObjectCreation() {
    // This test demonstrates that McdcCoverage objects are only created when requested
    McdcCoverageData data = McdcCoverageData.create();
    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(20, 3, 'f', 3, 1, "c || d");

    // Each call to getAllRecords creates new objects
    List<McdcCoverage> records1 = data.getAllRecords();
    List<McdcCoverage> records2 = data.getAllRecords();

    // Different object instances but same data
    assertThat(records1).isNotSameInstanceAs(records2);
    assertThat(records1.get(0)).isNotSameInstanceAs(records2.get(0));
    
    // But equivalent data
    assertThat(records1.get(0).lineNumber()).isEqualTo(records2.get(0).lineNumber());
    assertThat(records1.get(0).taken()).isEqualTo(records2.get(0).taken());
  }

  @Test
  public void testNrMcdcHit_countsOnlyNonZeroTaken() {
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");  // Hit
    data.addMcdc(10, 2, 'f', 0, 1, "a && b");  // Not hit
    data.addMcdc(20, 3, 't', 0, 0, "c || d");  // Not hit
    data.addMcdc(30, 1, 't', 1, 0, "e");       // Hit

    assertThat(data.size()).isEqualTo(4);
    assertThat(data.nrMcdcHit()).isEqualTo(2); // Only 2 were hit (taken > 0)
  }

  @Test
  public void testMergingZeroTakenRecords() {
    // Tests that merging two records with zero taken stays zero
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 0, 0, "a && b");
    data.addMcdc(10, 2, 't', 0, 0, "a && b"); // Same key, both zero

    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).taken()).isEqualTo(0);
    assertThat(records.get(0).wasHit()).isFalse();
  }

  @Test
  public void testMergingWithZeroAndNonZero() {
    // Tests that merging zero with non-zero produces non-zero
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 0, 0, "a && b");
    data.addMcdc(10, 2, 't', 5, 0, "a && b"); // Same key

    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).taken()).isEqualTo(5);
    assertThat(records.get(0).wasHit()).isTrue();
  }

  @Test
  public void testLargeDataSet_performanceAndCorrectness() {
    // Tests that the data structure handles a large number of records correctly
    McdcCoverageData data = McdcCoverageData.create();
    int numRecords = 1000;

    // Add many unique records
    for (int i = 0; i < numRecords; i++) {
      int line = i / 10; // 10 records per line
      int index = i % 10;
      data.addMcdc(line, 10, i % 2 == 0 ? 't' : 'f', i, index, "expr_" + line);
    }

    assertThat(data.size()).isEqualTo(numRecords);
    
    // Verify all are retrievable
    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records).hasSize(numRecords);
  }

  @Test
  public void testWhitespaceInExpression() {
    // Tests that expressions with whitespace are handled correctly
    McdcCoverageData data = McdcCoverageData.create();

    data.addMcdc(10, 2, 't', 5, 0, "a && b");
    data.addMcdc(10, 2, 't', 3, 0, "a&&b"); // No whitespace - different key

    assertThat(data.size()).isEqualTo(2); // Different expressions
  }

  @Test
  public void testSpecialCharactersInExpression() {
    McdcCoverageData data = McdcCoverageData.create();
    String expr = "'a' in 'a && b' || (c != \"test\")";

    data.addMcdc(10, 3, 't', 5, 0, expr);

    List<McdcCoverage> records = data.getAllRecords();
    assertThat(records.get(0).expression()).isEqualTo(expr);
  }

  @Test
  public void testAllFieldsCombinationUniqueness() {
    // Comprehensive test that changing any single field creates a different key
    McdcCoverageData data = McdcCoverageData.create();
    int baseLineNumber = 10;
    int baseGroupSize = 2;
    char baseSense = 't';
    int baseIndex = 0;
    String baseExpr = "base";

    // Base record
    data.addMcdc(baseLineNumber, baseGroupSize, baseSense, 1, baseIndex, baseExpr);

    // Change each field individually
    data.addMcdc(baseLineNumber + 1, baseGroupSize, baseSense, 1, baseIndex, baseExpr);
    data.addMcdc(baseLineNumber, baseGroupSize + 1, baseSense, 1, baseIndex, baseExpr);
    data.addMcdc(baseLineNumber, baseGroupSize, 'f', 1, baseIndex, baseExpr);
    data.addMcdc(baseLineNumber, baseGroupSize, baseSense, 1, baseIndex + 1, baseExpr);
    data.addMcdc(baseLineNumber, baseGroupSize, baseSense, 1, baseIndex, baseExpr + "_diff");

    // Should have 6 unique records
    assertThat(data.size()).isEqualTo(6);
  }

  @Test
  public void testCopyPreservesAllData() {
    McdcCoverageData original = McdcCoverageData.create();
    original.addMcdc(10, 2, 't', 5, 0, "a && b");
    original.addMcdc(20, 3, 'f', 3, 1, "c || d");
    original.addMcdc(30, 1, 't', 0, 0, "e"); // Zero taken

    McdcCoverageData copy = McdcCoverageData.copy(original);

    assertThat(copy.size()).isEqualTo(original.size());
    assertThat(copy.nrMcdcHit()).isEqualTo(original.nrMcdcHit());
    
    List<McdcCoverage> originalRecords = original.getAllRecords();
    List<McdcCoverage> copyRecords = copy.getAllRecords();
    
    // Same number and values
    assertThat(copyRecords).hasSize(originalRecords.size());
  }
}
