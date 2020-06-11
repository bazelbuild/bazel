/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.covariantreturn;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/** Test source for {@link NioBufferRefConverterTest}. */
public final class NioBufferInvocations {

  public static IntBuffer getIntBufferPosition(IntBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static CharBuffer getCharBufferPosition(CharBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static FloatBuffer getFloatBufferPosition(FloatBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static DoubleBuffer getDoubleBufferPosition(DoubleBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static ShortBuffer getShortBufferPosition(ShortBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static LongBuffer getLongBufferPosition(LongBuffer buffer, int position) {
    return buffer.position(position);
  }

  public static ByteBuffer getByteBufferPosition(ByteBuffer buffer, int position) {
    return buffer.position(position);
  }

  private NioBufferInvocations() {}
}
