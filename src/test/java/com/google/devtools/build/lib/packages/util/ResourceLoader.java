// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.ByteStreams;
import java.io.IOException;
import java.io.InputStream;

/**
 * Reads a file from the resources and returns its contents as a string.
 */
public final class ResourceLoader {
  /**
   * Reads a file from the resources and returns its contents as a string.
   */
  public static String readFromResources(String filename) throws IOException {
    InputStream in = ResourceLoader.class.getClassLoader().getResourceAsStream(filename);
    return new String(ByteStreams.toByteArray(in), UTF_8);
  }
}
