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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.util.Pair;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Task for tokenizing the contents of the {@link }ByteBufferFragment}
 * (with underlying {@link ByteBuffer}).
 * Intended to be called in parallel for the fragments of the {@link ByteBuffer} for lexing the
 * contents into independent logical tokens.
 *
 * {@link ParallelFileProcessing}
 */
public class BufferTokenizer implements Callable<List<Pair<Integer, ByteBufferFragment>>> {
  private final ByteBufferFragment bufferFragment;
  private final TokenConsumer consumer;
  private final SeparatorPredicate separatorPredicate;
  private final int offset;
  private final List<Pair<Integer, ByteBufferFragment>> fragments;

  /**
   * @param bufferFragment {@link ByteBufferFragment}, fragment of which should be tokenized
   * @param consumer token consumer
   * @param separatorPredicate predicate for separating tokens
   * @param offset start offset of <code>buffer</code> from the beginning of the file
   */
  public BufferTokenizer(ByteBufferFragment bufferFragment,
      TokenConsumer consumer,
      SeparatorPredicate separatorPredicate,
      int offset) {
    this.bufferFragment = bufferFragment;
    this.consumer = consumer;
    this.separatorPredicate = separatorPredicate;
    this.offset = offset;
    fragments = Lists.newArrayList();
  }

  /**
   * Returns the list of pairs (offset from the beginning of the file, fragment) of the
   * fragments on the bounds of the current fragment, which should be potentially merged with
   * fragments from the neighbor buffer fragments.
   *
   * Combined list of such fragments is passed to {@link TokenAssembler} for merging.
   */
  @Override
  public List<Pair<Integer, ByteBufferFragment>> call() throws Exception {
    int start = 0;
    for (int i = 0; i < bufferFragment.length() - 2; i++) {
      byte previous = bufferFragment.byteAt(i);
      byte current = bufferFragment.byteAt(i + 1);
      byte next = bufferFragment.byteAt(i + 2);

      if (!separatorPredicate.test(previous, current, next)) {
        continue;
      }
      ByteBufferFragment fragment = bufferFragment.subFragment(start, i + 2);
      if (start > 0) {
        consumer.token(fragment);
      } else {
        addFragment(fragment);
      }
      start = i + 2;
    }
    // There is always at least one byte at the bounds of the fragment.
    addFragment(bufferFragment.subFragment(start, bufferFragment.length()));
    return fragments;
  }

  private void addFragment(ByteBufferFragment fragment) {
    fragments.add(Pair.of(offset + fragment.getStartIncl(), fragment));
  }
}
