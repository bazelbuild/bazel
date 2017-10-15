// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An ActionInput that does not actually exist on the filesystem, but can still be written to an
 * OutputStream.
 */
public interface VirtualActionInput extends ActionInput {
  /**
   * Writes the fake file to an OutputStream. MUST be deterministic, in that multiple calls to write
   * the same VirtualActionInput must write identical bytes.
   */
  void writeTo(OutputStream out) throws IOException;

  /**
   * Gets a {@link ByteString} representation of the fake file. Used to avoid copying if the fake
   * file is internally represented as a {@link ByteString}.
   */
  ByteString getBytes() throws IOException;
}
