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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the {@link ThreadDumpAnalyzer} class. */
@RunWith(JUnit4.class)
public final class ThreadDumpAnalyzerTest {
  @Test
  public void analyze_groupsThreadsWithSameStackTrace() throws Exception {
    String input =
        """
        #1 "Thread 1"
            at Test.foo(Test.java:1)
            at Test.bar(Test.java:2)

        #2 "Thread 2"
            at Test.baz(Test.java:1)

        #3 "Thread 3"
            at Test.foo(Test.java:1)
            at Test.bar(Test.java:2)

        """;

    String output = analyze(input);

    assertThat(output)
        .isEqualTo(
            """
            #1 "Thread 1"
            #3 "Thread 3"
                at Test.foo(Test.java:1)
                at Test.bar(Test.java:2)

            #2 "Thread 2"
                at Test.baz(Test.java:1)

            """);
  }

  @Test
  public void analyze_sortsThreadsByName() throws Exception {
    String input =
        """
        #1 "Thread 4"
            at Test.foo(Test.java:1)
            at Test.bar(Test.java:2)

        #2 "Thread 2"
            at Test.baz(Test.java:1)

        #3 "Thread 3"
            at Test.foo(Test.java:1)
            at Test.bar(Test.java:2)

        """;

    String output = analyze(input);

    assertThat(output)
        .isEqualTo(
            """
            #2 "Thread 2"
                at Test.baz(Test.java:1)

            #3 "Thread 3"
            #1 "Thread 4"
                at Test.foo(Test.java:1)
                at Test.bar(Test.java:2)

            """);
  }

  @Test
  public void analyze_groupsThreadsWithEmptyStackTrace() throws Exception {
    String input =
        """
        #1 "Thread 1"

        #2 "Thread 2"
            at Test.baz(Test.java:1)

        #3 "Thread 3"

        """;

    String output = analyze(input);

    assertThat(output)
        .isEqualTo(
            """
            #1 "Thread 1"
            #3 "Thread 3"

            #2 "Thread 2"
                at Test.baz(Test.java:1)

            """);
  }

  @Test
  public void analyze_keepsNonThreadLines() throws Exception {
    String input =
        """
        #1 "Thread 1"

        #2 "Thread 2"

        foo
            bar

        #3 "Thread 3"

        """;

    String output = analyze(input);

    assertThat(output)
        .isEqualTo(
            """
            foo
                bar

            #1 "Thread 1"
            #2 "Thread 2"
            #3 "Thread 3"

            """);
  }

  @Test
  public void analyze_groupsThreadsWithSameStackTraceButDifferentStates() throws Exception {
    String input =
        """
        #1 "Thread 1" WAITING
            at Test.foo(Test.java:1)
            - waiting on <Object@1>
            at Test.bar(Test.java:2)

        #2 "Thread 2" RUNNABLE
            at Test.foo(Test.java:1)
            - locked <Object@1>
            at Test.bar(Test.java:2)

        #3 "Thread 3" RUNNABLE
            at Test.baz(Test.java:1)

        """;

    String output = analyze(input);

    assertThat(output)
        .isEqualTo(
            """
            #1 "Thread 1" WAITING
                - waiting on <Object@1>
            #2 "Thread 2" RUNNABLE
                - locked <Object@1>
                at Test.foo(Test.java:1)
                at Test.bar(Test.java:2)

            #3 "Thread 3" RUNNABLE
                at Test.baz(Test.java:1)

            """);
  }

  private static String analyze(String input) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ThreadDumpAnalyzer analyzer = new ThreadDumpAnalyzer();
    analyzer.analyze(new ByteArrayInputStream(input.getBytes(UTF_8)), out);
    return out.toString(UTF_8);
  }
}
