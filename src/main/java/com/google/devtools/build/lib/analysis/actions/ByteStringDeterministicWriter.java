// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.actions;

import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction.DeterministicWriter;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/**
 * A {@link DeterministicWriter} that wraps a {@link ByteString}. Use to avoid {@link ByteString}
 * copies.
 */
public class ByteStringDeterministicWriter implements DeterministicWriter {
  private final ByteString byteString;

  public ByteStringDeterministicWriter(ByteString byteString) {
    this.byteString = byteString;
  }

  @Override
  public void writeOutputFile(OutputStream out) throws IOException {
    byteString.writeTo(out);
  }

  @Override
  public ByteString getBytes() throws IOException {
    return byteString;
  }
}
