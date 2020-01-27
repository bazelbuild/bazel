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
//

package com.google.devtools.build.lib.bazel.rules.ninja.file;

/**
 * Represents a fragment of file as a {@link ByteBufferFragment} of {@link java.nio.ByteBuffer},
 * starting at {@link #offset}. The first byte should be read at {@link #getRealStartOffset()}, the
 * length of fragment is {@link ByteBufferFragment#length()}.
 */
public class ByteFragmentAtOffset {
  private final int offset;
  private final ByteBufferFragment fragment;

  public ByteFragmentAtOffset(int offset, ByteBufferFragment fragment) {
    this.offset = offset;
    this.fragment = fragment;
  }

  public int getOffset() {
    return offset;
  }

  public int getRealStartOffset() {
    return offset + fragment.getStartIncl();
  }

  public ByteBufferFragment getFragment() {
    return fragment;
  }
}
