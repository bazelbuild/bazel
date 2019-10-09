// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.io.Files;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ParallelFileProcessing;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.file.ParallelFileProcessing}.
 *
 * {@link #doPerformanceTest(int)} sample results for the "parallel buffers processing" solution
 * motivation, showing 25-40% improvement over straightforward implementation:
 *
 * File size: 197,950 KB
 * Parallel run: 615 ms (615 ms, 640 ms, 589 ms)
 * Files.readLines() run: 1049 ms (982 ms, 1241 ms, 923 ms)
 *
 * File size: 97,669 KB
 * Parallel run: 303 ms (293 ms, 320 ms, 296 ms)
 * Files.readLines() run: 395 ms (434 ms, 361 ms, 389 ms)
 *
 * File size: 24,949 KB
 * Parallel run: 103 ms (100 ms, 105 ms, 103 ms)
 * Files.readLines() run: 94 ms (107 ms, 87 ms, 89 ms)
 */
@RunWith(JUnit4.class)
public class ParallelFileProcessingTest {
  @Test
  public void testVariant1() throws Exception {
    int limit = 2000;
    int blockSize = -1;
    doTestNumbers(limit, blockSize);
  }

  @Test
  public void testVariant2() throws Exception {
    int limit = 4000;
    int blockSize = 100;
    doTestNumbers(limit, blockSize);
  }

  @Test
  public void testVariant3() throws Exception {
    int limit = 500;
    int blockSize = 1129;
    doTestNumbers(limit, blockSize);
  }

  @Test
  public void testPerformanceMedium() throws Exception {
    doPerformanceTest(500);
  }

  private void doPerformanceTest(int limit) throws Exception {
    File file = randomFile(new Random(), limit);
    try {
      long[] parallel = nTimesAvg(() -> {
        List<CharSequence> lines = Collections.synchronizedList(Lists.newArrayList());
        ParallelFileProcessing.processFile(file.toPath(), lines::add,
            NinjaSeparatorPredicate.INSTANCE, -1, -1);
      }, 3);
      long[] usual = nTimesAvg(() -> {
        List<String> usualLines = Files.readLines(file, StandardCharsets.ISO_8859_1);
        assertThat(usualLines).isNotEmpty();
      }, 3);
      printPerformanceResults(file, parallel, usual);
    } finally {
      file.delete();
    }
  }

  private void printPerformanceResults(File file, long[] parallel, long[] usual) {
    System.out.println(
        String.format("\nFile size: %,d KB ", file.length()/1024) +
            "\nParallel run: " + printTimes(parallel) +
            "\nFiles.readLines() run: " + printTimes(usual));
  }

  @Test
  public void testPerformanceLarge() throws Exception {
    doPerformanceTest(2000);
  }

  @Test
  public void testPerformanceVeryLarge() throws Exception {
    doPerformanceTest(4000);
  }

  private String printTimes(long[] times) {
    List<String> values = Lists.newArrayList();
    double avg = 0;
    for (long time : times) {
      avg += ((double) time)/times.length;
      values.add(time + " ms");
    }
    return Math.round(avg) + " ms (" + String.join(", ", values) + ")";
  }

  private long[] nTimesAvg(Callback r, int num) throws Exception {
    long[] result = new long[num];
    for (int i = 0; i < num; i++) {
      long start = System.currentTimeMillis();
      r.process();
      long end = System.currentTimeMillis();
      result[i] = end - start;
    }
    return result;
  }

  private File randomFile(Random r, int limit) throws IOException {
    String[] strings = new String[limit];
    IntStream.range(0, limit).parallel()
        .forEach(i -> {
          StringBuilder sb = new StringBuilder();
          int len = 100 + r.nextInt(10000);
          for (int j = 0; j < len; j++) {
            sb.append(r.nextInt());
            int value = r.nextInt(50);
            if (value == 5) {
              sb.append('\n');
            }
          }

          strings[i] = sb.toString();
        });
    File file = File.createTempFile("test", ".txt");
    Files.write(String.join("\n", strings).getBytes(StandardCharsets.ISO_8859_1), file);
    return file;
  }

  private void doTestNumbers(int limit, int blockSize)
      throws IOException, GenericParsingException, ExecutionException, InterruptedException {
    File file = writeTestFile(limit);
    try {
      List<String> lines = Collections.synchronizedList(Lists.newArrayListWithCapacity(limit));
      ParallelFileProcessing.processFile(file.toPath(), s -> lines.add(s.toString()),
          NinjaSeparatorPredicate.INSTANCE, blockSize, -1);
      // Copy to non-synchronized list for check
      assertNumbers(limit, Lists.newArrayList(lines));
    } finally {
      file.delete();
    }
  }

  private void assertNumbers(int limit, List<String> lines) {
    Set<Integer> numbers = Sets.newHashSet();
    lines.forEach(s -> {
      boolean added = numbers.add(Integer.parseInt(s.trim()));
      assertThat(added).isTrue();
    });
    assertThat(numbers).hasSize(limit);
    for (int i = 0; i < limit; i++) {
      boolean removed = numbers.remove(i);
      assertThat(removed).isTrue();
    }
  }

  private File writeTestFile(int limit) throws IOException {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < limit; i++) {
      if (sb.length() > 0) {
        sb.append('\n');
      }
      sb.append(i);
    }
    File file = File.createTempFile("test", ".txt");
    Files.write(sb.toString().getBytes(StandardCharsets.ISO_8859_1), file);
    return file;
  }

  private interface Callback {
    void process() throws Exception;
  }
}
