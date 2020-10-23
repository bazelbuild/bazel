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
package com.google.devtools.build.skydoc.renderer;

import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Helper to handle Proto file I/O. This abstraction is useful for tests which don't involve actual
 * file I/O.
 */
public interface ProtoFileAccessor {
  /**
   * Returns the bytes from the raw proto file.
   *
   * @param inputPathString the path of the input raw {@link StardocOutputProtos} file.
   */
  byte[] getProtoContent(String inputPathString) throws IOException;

  /** Returns true if a file exists at the current path. */
  boolean fileExists(String pathString);

  /**
   * Creates a {@link FileOutputStream} and writes the bytes to the output location.
   *
   * @param outputPathString the output location that is being written to
   * @param content the bytes from input proto file
   */
  void writeToOutputLocation(String outputPathString, byte[] content) throws IOException;
}
