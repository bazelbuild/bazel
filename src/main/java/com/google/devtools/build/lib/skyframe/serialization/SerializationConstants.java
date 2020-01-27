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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.util.ResourceUsage;

/**
 * Some static constants for deciding serialization behavior.
 */
public class SerializationConstants {

  /** Number of threads in deserialization pools. */
  public static final int DESERIALIZATION_POOL_SIZE = 2 * ResourceUsage.getAvailableProcessors();

  private static final boolean IN_TEST = System.getenv("TEST_TMPDIR") != null;
  private static final boolean CHECK_SERIALIZATION =
      System.getenv("DONT_SANITY_CHECK_SERIALIZATION") == null;

  /**
   * Returns true if serialization should be validated on all Skyframe writes.
   */
  public static boolean shouldCheckSerializationBecauseInTest() {
    return IN_TEST && CHECK_SERIALIZATION;
  }
}
