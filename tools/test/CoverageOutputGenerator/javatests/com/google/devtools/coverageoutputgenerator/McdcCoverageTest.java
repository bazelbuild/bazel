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
import static org.junit.Assert.assertThrows;

import com.google.common.base.VerifyException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link McdcCoverage}. */
@RunWith(JUnit4.class)
public class McdcCoverageTest {

  @Test
  public void testCreate() {
    McdcCoverage mcdc = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");

    assertThat(mcdc.lineNumber()).isEqualTo(10);
    assertThat(mcdc.groupSize()).isEqualTo(2);
    assertThat(mcdc.sense()).isEqualTo('t');
    assertThat(mcdc.taken()).isEqualTo(5);
    assertThat(mcdc.index()).isEqualTo(0);
    assertThat(mcdc.expression()).isEqualTo("'a' in 'a && b'");
    assertThat(mcdc.wasHit()).isTrue();
  }

  @Test
  public void testCreateWithZeroTaken() {
    McdcCoverage mcdc = McdcCoverage.create(10, 1, 'f', 0, 0, "!b");

    assertThat(mcdc.lineNumber()).isEqualTo(10);
    assertThat(mcdc.groupSize()).isEqualTo(1);
    assertThat(mcdc.sense()).isEqualTo('f');
    assertThat(mcdc.taken()).isEqualTo(0);
    assertThat(mcdc.index()).isEqualTo(0);
    assertThat(mcdc.expression()).isEqualTo("!b");
    assertThat(mcdc.wasHit()).isFalse();
  }

  @Test
  public void testMergeIdenticalRecords() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 0, "'a' in 'a && b'");

    McdcCoverage merged = McdcCoverage.merge(mcdc1, mcdc2);

    assertThat(merged.lineNumber()).isEqualTo(10);
    assertThat(merged.groupSize()).isEqualTo(2);
    assertThat(merged.sense()).isEqualTo('t');
    assertThat(merged.taken()).isEqualTo(8);
    assertThat(merged.index()).isEqualTo(0);
    assertThat(merged.expression()).isEqualTo("'a' in 'a && b'");
    assertThat(merged.wasHit()).isTrue();
  }

  @Test
  public void testMergeZeroTaken() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 0, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 0, 0, "'a' in 'a && b'");

    McdcCoverage merged = McdcCoverage.merge(mcdc1, mcdc2);

    assertThat(merged.lineNumber()).isEqualTo(10);
    assertThat(merged.groupSize()).isEqualTo(2);
    assertThat(merged.sense()).isEqualTo('t');
    assertThat(merged.taken()).isEqualTo(0);
    assertThat(merged.index()).isEqualTo(0);
    assertThat(merged.expression()).isEqualTo("'a' in 'a && b'");
    assertThat(merged.wasHit()).isFalse();
  }

  @Test
  public void testMergeWithOneHitOneUnhit() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 0, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");

    McdcCoverage merged = McdcCoverage.merge(mcdc1, mcdc2);

    assertThat(merged.lineNumber()).isEqualTo(10);
    assertThat(merged.groupSize()).isEqualTo(2);
    assertThat(merged.sense()).isEqualTo('t');
    assertThat(merged.taken()).isEqualTo(5);
    assertThat(merged.index()).isEqualTo(0);
    assertThat(merged.expression()).isEqualTo("'a' in 'a && b'");
    assertThat(merged.wasHit()).isTrue();
  }

  @Test
  public void testDifferentLineNumbersFail() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(11, 2, 't', 3, 0, "'a' in 'a && b'");

    assertThrows(VerifyException.class, () -> McdcCoverage.merge(mcdc1, mcdc2));
  }

  @Test
  public void testDifferentGroupSizesFail() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 3, 't', 3, 0, "'a' in 'a && b'");

    assertThrows(VerifyException.class, () -> McdcCoverage.merge(mcdc1, mcdc2));
  }

  @Test
  public void testDifferentSensesFail() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 'f', 3, 0, "'a' in 'a && b'");

    assertThrows(VerifyException.class, () -> McdcCoverage.merge(mcdc1, mcdc2));
  }

  @Test
  public void testDifferentIndicesFail() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 1, "'b' in 'a && b'");

    assertThrows(VerifyException.class, () -> McdcCoverage.merge(mcdc1, mcdc2));
  }

  @Test
  public void testDifferentExpressionsFail() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 1, "'d' in 'c || d'");

    assertThrows(VerifyException.class, () -> McdcCoverage.merge(mcdc1, mcdc2));
  }

  @Test
  public void testEmptyExpression() {
    McdcCoverage mcdc = McdcCoverage.create(10, 0, 't', 1, 0, "");

    assertThat(mcdc.expression()).isEmpty();
    assertThat(mcdc.wasHit()).isTrue();
  }

  @Test
  public void testComplexExpression() {
    String complexExpr = "'d' in '((a || b) && (c || d)) || e'";
    McdcCoverage mcdc = McdcCoverage.create(50, 5, 'f', 10, 3, complexExpr);

    assertThat(mcdc.expression()).isEqualTo(complexExpr);
    assertThat(mcdc.lineNumber()).isEqualTo(50);
    assertThat(mcdc.groupSize()).isEqualTo(5);
    assertThat(mcdc.sense()).isEqualTo('f');
    assertThat(mcdc.taken()).isEqualTo(10);
    assertThat(mcdc.index()).isEqualTo(3);
    assertThat(mcdc.wasHit()).isTrue();
  }

  @Test
  public void testCanMergeIdenticalRecords() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 0, "'a' in 'a && b'");

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isTrue();
  }

  @Test
  public void testCanMergeDifferentLineNumbers() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(11, 2, 't', 3, 0, "'a' in 'a && b'");

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isFalse();
  }

  @Test
  public void testCanMergeDifferentGroupSizes() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 3, 't', 3, 0, "'a' in 'a && b'");

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isFalse();
  }

  @Test
  public void testCanMergeDifferentSenses() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 'f', 3, 0, "'a' in 'a && b'");

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isFalse();
  }

  @Test
  public void testCanMergeDifferentIndices() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 1, "'a' in 'a && b'");  // Expresion left wrong on purpose

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isFalse();
  }

  @Test
  public void testCanMergeDifferentExpressions() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 0, "'c' in 'c || d'");

    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isFalse();
  }

  @Test
  public void testCanMergeDifferentTakenCounts() {
    McdcCoverage mcdc1 = McdcCoverage.create(10, 2, 't', 5, 0, "'a' in 'a && b'");
    McdcCoverage mcdc2 = McdcCoverage.create(10, 2, 't', 3, 0, "'a' in 'a && b'");

    // Should be able to merge even with different taken counts
    assertThat(McdcCoverage.canMerge(mcdc1, mcdc2)).isTrue();
  }
}
