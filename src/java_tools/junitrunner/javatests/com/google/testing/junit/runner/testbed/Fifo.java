// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.testbed;

import com.google.common.io.ByteStreams;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;

/**
 * Utility class to synchronize test bed tests and shell tests using a FIFO file.
 */
public final class Fifo {
  private static final String FIFO = System.getProperty("test.fifo");

  private Fifo(){}

  /**
   * Helper method to help with the synchronization between testbed java test and shell tests. It
   * will block until data is available on the FIFO
   */
  static void waitUntilDataAvailable() throws IOException {
    if (FIFO == null) {
      throw new IllegalStateException("No fifo specified");
    }
    Files.asByteSource(new File(FIFO)).copyTo(ByteStreams.nullOutputStream());
  }
}
