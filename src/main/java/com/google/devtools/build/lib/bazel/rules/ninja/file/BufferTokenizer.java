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
import java.nio.ByteBuffer;
import java.util.List;
import java.util.ListIterator;
import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * Task for tokenizing the contents of the {@link }ByteBufferFragment}
 * (with underlying {@link ByteBuffer}).
 * Intended to be called in parallel for the fragments of the {@link }ByteBuffer} for lexing the
 * contents into independent logical tokens.
 *
 * {@link ParallelFileProcessing}
 */
public class BufferTokenizer implements Callable<Void> {
  private final ByteBufferFragment bufferFragment;
  private final TokenConsumer consumer;
  private final SeparatorPredicate separatorPredicate;
  private final int offset;
  private final List<ByteBufferFragment> fragments;

  /**
   * @param buffer buffer, fragment of which should be tokenized
   * @param consumer token consumer
   * @param separatorPredicate predicate for separating tokens
   * @param offset start offset of <code>buffer</code> from the beginning of the file
   * @param startIncl start index of the buffer fragment, inclusive
   * @param endExcl end index of the buffer fragment, exclusive, or the size of the buffer,
   */
  public BufferTokenizer(ByteBuffer buffer,
      TokenConsumer consumer,
      SeparatorPredicate separatorPredicate,
      int offset, int startIncl, int endExcl) {
    bufferFragment = new ByteBufferFragment(buffer, startIncl, endExcl);
    this.consumer = consumer;
    this.separatorPredicate = separatorPredicate;
    this.offset = offset;
    fragments = Lists.newArrayList();
  }

  @Override
  public Void call() throws Exception {
    int start = 0;
    ListIterator<Byte> iterator = bufferFragment.iterator();
    while (iterator.hasNext()) {
      Optional<Boolean> isSeparator = separatorPredicate.isSeparator(iterator.next(), iterator);
      if (isSeparator.isPresent() && !isSeparator.get()) {
        continue;
      }
      ByteBufferFragment fragment = bufferFragment.subFragment(start, iterator.nextIndex());
      if (isSeparator.isPresent() && start > 0) {
        consumer.token(fragment);
      } else {
        fragments.add(fragment);
      }
      start = iterator.nextIndex();
    }
    if (start < bufferFragment.length()) {
      fragments.add(bufferFragment.subFragment(start, bufferFragment.length()));
    }
    return null;
  }

  /**
   * @return start offset of the underlying ByteBuffer from the beginning of the file
   */
  public int getOffset() {
    return offset;
  }

  /**
   * @return fragments on the bounds of the buffer fragment, where we can not say if it is
   * a seaparate token, or a part of a bigger token.
   * They should be observed together with neighbour fragments from other BufferTokenizers.
   * {@link TokenAssembler} is solving this problem.
   */
  public List<ByteBufferFragment> getFragments() {
    return fragments;
  }
}
