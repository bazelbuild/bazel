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
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
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
 * motivation, showing significant improvement over straightforward implementation:
 *
 * File size: 1,321,144 KB
 * Parallel run: 727 ms (704 ms, 738 ms, 739 ms)
 * Files.readLines() run: 7344 ms (7888 ms, 7244 ms, 6900 ms)
 *
 * File size: 959,521 KB
 * Parallel run: 524 ms (539 ms, 546 ms, 486 ms)
 * Files.readLines() run: 5520 ms (5161 ms, 5350 ms, 6048 ms)
 *
 * File size: 24,872 KB
 * Parallel run: 86 ms (124 ms, 79 ms, 56 ms)
 * Files.readLines() run: 113 ms (136 ms, 112 ms, 91 ms)
 *
 * File size: 100,184 KB
 * Parallel run: 74 ms (75 ms, 77 ms, 69 ms)
 * Files.readLines() run: 418 ms (540 ms, 360 ms, 354 ms)
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

  private static void doPerformanceTest(int limit) throws Exception {
    File file = randomFile(new Random(), limit);
    try {
      // Currently we do not call toString() method, as it reduces performance in X times;
      // However, further parsing / conversion to string can be done differently
      // (for instance, we could decode the underlying big byte buffers and read
      // corresponding decoded parts from there)
      long[] parallel = nTimesAvg(() -> {
        List<List<ByteBufferFragment>> list = Lists.newArrayList();
        ParallelFileProcessing.processFile(file.toPath(),
            () -> {
              List<ByteBufferFragment> inner = Lists.newArrayList();
              list.add(inner);
              return inner::add;
            },
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

  private static void printPerformanceResults(File file, long[] parallel, long[] usual) {
    System.out.println(
        String.format("\nFile size: %,d KB ", file.length()/1024) +
            "\nParallel run: " + printTimes(parallel) +
            "\nFiles.readLines() run: " + printTimes(usual));
  }

  private static String printTimes(long[] times) {
    List<String> values = Lists.newArrayList();
    double avg = 0;
    for (long time : times) {
      avg += ((double) time)/times.length;
      values.add(time + " ms");
    }
    return Math.round(avg) + " ms (" + String.join(", ", values) + ")";
  }

  private static long[] nTimesAvg(Callback r, int num) throws Exception {
    long[] result = new long[num];
    for (int i = 0; i < num; i++) {
      long start = System.currentTimeMillis();
      r.process();
      long end = System.currentTimeMillis();
      result[i] = end - start;
    }
    return result;
  }

  private static File randomFile(Random r, int limit) throws IOException {
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
      ParallelFileProcessing.processFile(file.toPath(), () -> s -> lines.add(s.toString()),
          NinjaSeparatorPredicate.INSTANCE, blockSize, -1);
      // Copy to non-synchronized list for check
      assertNumbers(limit, Lists.newArrayList(lines));
    } finally {
      file.delete();
    }
  }

  private static void assertNumbers(int limit, List<String> lines) {
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

  private static File writeTestFile(int limit) throws IOException {
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
