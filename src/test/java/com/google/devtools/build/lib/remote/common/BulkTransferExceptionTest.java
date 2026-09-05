// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.common;

import static com.google.common.truth.Truth.assertThat;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BulkTransferExceptionTest {

  private static String stackTraceAsString(Throwable t) {
    StringWriter sw = new StringWriter();
    t.printStackTrace(new PrintWriter(sw));
    return sw.toString();
  }

  @Test
  public void shouldProvideGenericMessageIfNoAddedException() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldProvideGenericMessageIfOnlyNullMessages() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException());
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldReturnStackTraceFromSingleException() {
    IOException cause = new IOException("Failure Type A");
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(cause);
    String message = bulkTransferException.getMessage();
    assertThat(message).isEqualTo("1 errors during bulk transfer (1 unique):\n\n" + stackTraceAsString(cause) + "\n\n");
  }

  @Test
  public void shouldDeduplicateExceptionsWithSameMessage() {
    IOException cause = new IOException("Failure Type A");
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(cause);
    bulkTransferException.add(cause);
    bulkTransferException.add(cause);
    String message = bulkTransferException.getMessage();
    assertThat(message).startsWith("3 errors during bulk transfer (1 unique):\n\n");
    assertThat(message.indexOf("Failure Type A")).isEqualTo(message.lastIndexOf("Failure Type A"));
  }

  @Test
  public void shouldIncludeAllDistinctStackTracesWhenAggregating() {
    IOException causeA = new IOException("Failure Type A");
    IOException causeB = new IOException("Failure Type B");
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(causeA);
    bulkTransferException.add(causeB);
    String message = bulkTransferException.getMessage();
    assertThat(message).startsWith("2 errors during bulk transfer (2 unique):\n\n");
    assertThat(message).contains(stackTraceAsString(causeA));
    assertThat(message).contains(stackTraceAsString(causeB));
  }

  @Test
  public void shouldIncludeCauseChainInStackTrace() {
    IOException rootCause = new IOException("root: UNAVAILABLE");
    IOException wrapping = new IOException("grpc: UNAVAILABLE", rootCause);
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(wrapping);
    String message = bulkTransferException.getMessage();
    assertThat(message).contains("grpc: UNAVAILABLE");
    assertThat(message).contains("root: UNAVAILABLE");
  }
}
