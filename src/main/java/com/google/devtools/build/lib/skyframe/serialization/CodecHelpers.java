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

/**
 * Helper methods for writing codecs.
 *
 * <p>Supports 16-bit types that not included in {@link CodedInputStream} and {@link
 * CodedOutputStream}.
 */
public final class CodecHelpers {
  public static void writeShort(CodedOutputStream codedOut, short value) throws IOException {
    codedOut.writeRawByte((byte) (value >> 8));
    codedOut.writeRawByte((byte) value);
  }

  public static short readShort(CodedInputStream codedIn) throws IOException {
    int buffer = codedIn.readRawByte() << 8;
    buffer |= (codedIn.readRawByte() & 0xFF);
    return (short) buffer;
  }

  public static void writeChar(CodedOutputStream codedOut, char value) throws IOException {
    codedOut.writeRawByte((byte) (value >> 8));
    codedOut.writeRawByte((byte) value);
  }

  public static char readChar(CodedInputStream codedIn) throws IOException {
    int buffer = codedIn.readRawByte() << 8;
    buffer |= (codedIn.readRawByte() & 0xFF);
    return (char) buffer;
  }

  private CodecHelpers() {} // Just a static method namespace. No instances.
}
