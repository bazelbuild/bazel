// Copyright 2023 The Bazel Authors. All rights reserved.
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


import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/** Helper methods for writing codecs. */
final class CodecHelpers {
  static void writeShort(CodedOutputStream codedOut, short value) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).putShort(value);
    codedOut.writeRawBytes(buffer);
  }

  static short readShort(CodedInputStream codedIn) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
    return buffer.getShort(0);
  }

  static void writeChar(CodedOutputStream codedOut, char value) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).putChar(value);
    codedOut.writeRawBytes(buffer);
  }

  static char readChar(CodedInputStream codedIn) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(2).put(codedIn.readRawBytes(2));
    return buffer.getChar(0);
  }

  private CodecHelpers() {} // Just a static method namespace. No instances.
}
