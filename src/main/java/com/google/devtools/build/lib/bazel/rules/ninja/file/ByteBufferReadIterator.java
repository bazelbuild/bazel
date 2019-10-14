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

import java.util.ListIterator;

/**
 * Read-only bidirectional iterator of the bytes in {@link ByteBufferFragment}.
 */
public class ByteBufferReadIterator implements ListIterator<Byte> {
  private final ByteBufferFragment fragment;
  private int position;

  ByteBufferReadIterator(ByteBufferFragment fragment) {
    this.fragment = fragment;
    position = 0;
  }

  ByteBufferReadIterator(ByteBufferFragment fragment, int position) {
    if (position < 0 || position > (fragment.length() + 1)) {
      throw new IndexOutOfBoundsException();
    }
    this.fragment = fragment;
    this.position = position;
  }

  @Override
  public boolean hasNext() {
    return position < fragment.length();
  }

  @Override
  public Byte next() {
    return fragment.byteAt(position++);
  }

  @Override
  public boolean hasPrevious() {
    return position > 0;
  }

  @Override
  public Byte previous() {
    return fragment.byteAt(--position);
  }

  @Override
  public int nextIndex() {
    return position;
  }

  @Override
  public int previousIndex() {
    return position - 1;
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void set(Byte aByte) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void add(Byte aByte) {
    throw new UnsupportedOperationException();
  }
}
