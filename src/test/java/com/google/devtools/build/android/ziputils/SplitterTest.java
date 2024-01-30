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
package com.google.devtools.build.android.ziputils;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.Range;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link Splitter}.
 */
@RunWith(JUnit4.class)
public class SplitterTest {

  private static final String ARCHIVE_DIR_SUFFIX = "/";
  private static final String ARCHIVE_FILE_SEPARATOR = "/";
  private static final String CLASS_SUFFIX = ".class";

  @Test
  public void testAssign() {
    int size = 10;

    Collection<String> input;
    ArrayList<String> filter;
    Map<String, Integer> output;

    input = genEntries(size, size);
    filter = new ArrayList<>(10);
    for (int i = 0; i < size; i++) {
      filter.add("dir" + i + ARCHIVE_FILE_SEPARATOR + "file" + i + CLASS_SUFFIX);
    }
    Splitter splitter = new Splitter(size + 1, input.size());
    splitter.assignAllToCurrentShard(filter);
    splitter.nextShard();
    output = new LinkedHashMap<>();
    for (String path : input) {
      output.put(path, splitter.assign(path));
    }

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        String path = "dir" + i + ARCHIVE_FILE_SEPARATOR + "file" + j + CLASS_SUFFIX;
        if (i == j) {
          assertWithMessage(path).that(output.get(path)).isEqualTo(0);
        } else {
          assertWithMessage(path).that(output.get(path)).isEqualTo(i + 1);
        }
      }
    }
  }


  /**
   * Test splitting of single-ile packages. Note, this is also testing for the situation
   * where input entries are unordered, and thus appearing to be in different packages,
   * to the implementation that only confiders the previous file to determine package
   * boundaries.
   */
  @Test
  public void testSingleFilePackages() {
    int[][] params = {
      { 1, 1, 1},  // one shard, for one package with one file
      {1, 2, 1},   // one shard, for two packages, with one file each
      {1, 10, 1},  // one shard, for ten packages, with one file each
      {2, 2, 1},   // ...
      {2, 10, 1},
      {2, 100, 1},
      {10, 10, 1},
      {10, 15, 1},
      {10, 95, 1},
      {97, 10000, 1},
    };
    comboRunner(params);
  }

  /**
   * Test cases where the number of shards is less than the number
   * of packages. This implies that the package size is less than
   * the average shard size. We expect shards to be multiple of
   * package size.
   */
  @Test
  public void testPackageSplit() {
    int[][] params = {
      {2, 3, 2},  // two shards, for three packages, with two files each
      {2, 3, 9},  // ...
      {2, 3, 10},
      {2, 3, 11},
      {2, 3, 19},

      {2, 10, 2},
      {2, 10, 9},
      {2, 10, 10},
      {2, 10, 11},
      {2, 10, 19},

      {10, 11, 2},
      {10, 11, 9},
      {10, 11, 10},
      {10, 11, 11},
      {10, 11, 19},

      {10, 111, 2},
      {10, 111, 9},
      {10, 111, 10},
      {10, 111, 11},
      {10, 111, 19},

      {25, 1000, 8},
      {25, 1000, 10},
      {25, 1000, 19},

      {250, 10000, 19},
    };
    comboRunner(params);
  }

  /**
   * Tests situations where the number of shards exceeds the number of
   * packages (but not the number of files). That is, the implementation
   * must split at least one package.
   */
  @Test
  public void testForceSplit() {
    int[][] params = {
      {2, 1, 2},  // two shards, for one package, with two files
      {2, 1, 9},  // ...
      {2, 1, 10},
      {2, 1, 11},

      {3, 2, 2},
      {10, 9, 2},
      {10, 2, 9},
      {10, 9, 9},
      {10, 2, 10},
      {10, 9, 10},
      {10, 2, 11},
      {10, 9, 11},
      {10, 2, 111},
      {10, 9, 111},

      {100, 12, 9},
      {100, 12, 9},
      {100, 10, 10},
      {100, 10, 10},
      {100, 10, 11},
      {100, 20, 111},
    };
    comboRunner(params);
  }

  /**
   * Tests situation where the number of shards requested exceeds the
   * the number of files. Empty shards are expected.
   */
  @Test
  public void testEmptyShards() {
    int[][] params = {
      {2, 1, 1},  // two shards, for one package, with one files
      {10, 2, 2},
      {100, 10, 9},
      {100, 9, 10},
    };
    comboRunner(params);
  }

  /**
   * Run multiple test for sets of test specifications consisting of
   * "number of shards", "number of packages", "package size".
   */
  private void comboRunner(int[][] params) {
    Collection<String> input;
    Map<String, Integer> output;

    for (int[] param : params) {
      input = genEntries(param[1], param[2]);
      output = runOne(param[0], input);
      String name = Arrays.toString(param);
      splitAsserts(name, param[0], param[1], param[2],
          commonAsserts(name, param[0], param[1], param[2], input, output));
    }
  }

  private Map<String, Integer> runOne(int shards, Collection<String> entries) {
    Splitter splitter = new Splitter(shards, entries.size());
    Map<String, Integer> result = new LinkedHashMap<>();
    for (String entry : entries) {
      result.put(entry, splitter.assign(entry));
    }
    return result;
  }

  private Collection<String> genEntries(int packages, int files) {
    List<String> entries = new ArrayList<>();
    for (int dir = 0; dir < packages; dir++) {
      for (int file = 0; file < files; file++) {
        entries.add("dir" + dir + ARCHIVE_FILE_SEPARATOR + "file" + file + CLASS_SUFFIX);
      }
    }
    return entries;
  }

  private int[] assertAndCountMappings(int shards, int packageSize,
    Map<String, Integer> output, boolean expectPackageBoundaryShards) {
    int[] counts = new int[shards + 1];
    String prevPath = null;
    int prev = -2;
    for (Map.Entry<String, Integer> entry : output.entrySet()) {
      String path = entry.getKey();
      int assignment = entry.getValue();
      assertThat(assignment).isIn(Range.closed(0, shards));
      counts[assignment + 1]++;
      if (path.endsWith(CLASS_SUFFIX)) {
        if (prev == -2) {
          assertThat(assignment).isEqualTo(0);
        } else if (prev > 0 && prev != assignment) {
          assertThat(assignment).isEqualTo(prev + 1); // shard index increasing
          if (expectPackageBoundaryShards) {
            String prevDir = prevPath.substring(0, prevPath.lastIndexOf(ARCHIVE_DIR_SUFFIX));
            String dir = path.substring(0, path.lastIndexOf(ARCHIVE_DIR_SUFFIX));
            // package boundary, or full packages
            assertThat(!prevDir.equals(dir) || counts[prev + 1] % packageSize != 0).isTrue();
          }
        }
        prevPath = path;
      }
      prev = assignment;
    }
    return counts;
  }

  /**
   * Validate that generated mapping maintains input order.
   */
  private void assertMaintainOrder(Collection<String> input, Map<String, Integer> output) {
    assertThat(output.keySet()).containsExactlyElementsIn(input).inOrder();
  }

  /**
   * Verifies that packages have not been unnecessarily split.
   */
  private void assertNoSplit(String name, int packageSize, int[] counts) {
    for (int i = 1; i < counts.length; i++) {
      assertWithMessage(name + " shard " + i).that(counts[i]).isAtLeast(0);
    }
  }

  /**
   * Verifies the presence of package-split in the tailing shards.
   */
  private void assertHasSplit(String name, int packageSize, int[] counts) {
    for (int i = 1; i < counts.length - 1; i++) {
      if (counts[i + 1] <= 1) {
        continue;
      }
      assertWithMessage(name + " shard " + i).that(counts[i]).isAtMost(packageSize);
    }
  }

  /**
   * Verify the presence of tailing empty shards, if unavoidable.
   */
  private void assertHasEmpty(String name, int[] counts, boolean expectEmpty) {
    boolean hasEmpty = false;
    for (int i = 1; i < counts.length; i++) {
      if (counts[i] == 0) {
        hasEmpty = true;
      } else {
        assertThat(!hasEmpty || counts[i] == 0).isTrue();
      }
    }
    assertWithMessage(name).that(hasEmpty).isEqualTo(expectEmpty);
  }

  /**
   * Validates that each shard meets expected minimal and maximum size requirements,
   * to ensure that shards are reasonably evenly sized.
   */
  private void assertBalanced(String name, int shards, int packageCount, int packageSize,
      int entries, int[] counts) {
    int classes = packageSize * packageCount;
    int noneClass = entries - counts[0] - classes;
    int idealSize = Math.max(1, classes / shards);
    int delta = Math.min(Math.min(10, (idealSize + 3) >> 2), (int) Math.log(shards));
    int lowerBound = idealSize - delta;
    int upperBound = idealSize + delta;
    for (int i = 1; i < counts.length; i++) {
      int adjusted = i == 1 ? counts[i] - noneClass : counts[i];
      if (i < shards && counts[i + 1] > 1) {
        if (shards <= packageCount) {
          // if there are fewer shards than packages, expect shards contain at least 1 full package
          assertWithMessage(name + " dense shard " + i)
              .that(counts[i])
              .isIn(Range.closed(packageSize, entries));
        } else {
          assertWithMessage(name + " sparse shard " + i)
              .that(counts[i])
              .isIn(Range.closed(0, packageSize));
        }
        if (noneClass == 0 && counts[0] == 0) {
          // Give some slack in minimal number of entries in a shard because Splitter recomputes
          // boundaries for each shard, so our computed bounds can be off for later shards.
          assertWithMessage(name + " shard " + i)
              .that(counts[i])
              .isIn(Range.closed(lowerBound - i, entries));
        }
      }
      // Give some slack in maximum number of entries in a shard because Splitter recomputes
      // boundaries for each shard, so our computed bounds can be off for later shards.
      assertWithMessage(name + " shard " + i).that(adjusted).isAtMost(upperBound + i);
    }
  }

  /**
   * Verifies that packages are only split as expected, and that no unexpected
   * empty shards are generated.
   */
  private void splitAsserts(String name, int shards, int packageCount, int packageSize,
      int[] counts) {
    boolean emptyExpected = packageCount * packageSize < shards;
    boolean splitExpected = shards > packageCount;
    if (splitExpected) {
      assertHasSplit(name, packageSize, counts);
    } else {
      assertNoSplit(name, packageSize, counts);
    }
    assertHasEmpty(name, counts, emptyExpected);
  }

  /**
   * Checks assert applicable to all tests.
   */
  private int[] commonAsserts(String name, int shards, int packageCount, int packageSize,
      Collection<String> input, Map<String, Integer> output) {
    assertMaintainOrder(input, output);
    int[] counts = assertAndCountMappings(shards, packageSize, output, packageCount <= shards);
    assertBalanced(name, shards, packageCount, packageSize, input.size(), counts);
    return counts;
  }
}
