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
import java.util.concurrent.Callable;

/**
 * Task for splitting the contents of the {@link ByteBufferFragment} (with underlying {@link
 * ByteBuffer}). Intended to be called in parallel for the fragments of the {@link ByteBuffer} for
 * lexing the contents into independent logical declarations.
 *
 * <p>{@link ParallelFileProcessing}
 */
public class BufferSplitter implements Callable<List<ByteFragmentAtOffset>> {
  private final ByteBufferFragment bufferFragment;
  private final DeclarationConsumer consumer;
  private final SeparatorFinder separatorFinder;
  private final int offset;

  /**
   * @param bufferFragment {@link ByteBufferFragment}, fragment of which should be splitted
   * @param consumer declaration consumer
   * @param separatorFinder finds declaration separators
   * @param offset start offset of <code>buffer</code> from the beginning of the file
   */
  public BufferSplitter(
      ByteBufferFragment bufferFragment,
      DeclarationConsumer consumer,
      SeparatorFinder separatorFinder,
      int offset) {
    this.bufferFragment = bufferFragment;
    this.consumer = consumer;
    this.separatorFinder = separatorFinder;
    this.offset = offset;
  }

  /**
   * Returns the list of {@link ByteFragmentAtOffset} - fragments on the bounds of the current
   * fragment, which should be potentially merged with fragments from the neighbor buffer fragments.
   *
   * <p>Combined list of {@link ByteFragmentAtOffset} is passed to {@link DeclarationAssembler} for
   * merging.
   */
  @Override
  public List<ByteFragmentAtOffset> call() throws Exception {
    List<ByteFragmentAtOffset> fragments = Lists.newArrayList();
    int start = 0;
    while (true) {
      int end = separatorFinder.findNextSeparator(bufferFragment, start, -1);
      if (end < 0) {
        break;
      }
      ByteBufferFragment fragment = bufferFragment.subFragment(start, end + 1);
      ByteFragmentAtOffset fragmentAtOffset = new ByteFragmentAtOffset(offset, fragment);
      if (start > 0) {
        consumer.declaration(fragmentAtOffset);
      } else {
        fragments.add(fragmentAtOffset);
      }
      start = end + 1;
    }
    // There is always at least one byte at the bounds of the fragment.
    ByteBufferFragment lastFragment = bufferFragment.subFragment(start, bufferFragment.length());
    fragments.add(new ByteFragmentAtOffset(offset, lastFragment));
    return fragments;
  }
}
