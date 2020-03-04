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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Range;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * A {@link BufferSplitter} callback interface implementation, that assembles fragments of
 * declarations (that may occur on the edges of byte buffer fragments) together and passes all
 * declarations to delegate {@link DeclarationConsumer}, which does further processing / parsing.
 */
public class DeclarationAssembler {
  private final DeclarationConsumer declarationConsumer;
  private final SeparatorFinder separatorFinder;

  /**
   * @param declarationConsumer delegate declaration consumer for actual processing / parsing
   * @param separatorFinder callback used to determine if two fragments should be separate
   *     declarations (in the Ninja case, if the new line starts with a space, it should be treated
   *     as a part of the previous declaration, i.e. the separator is longer then one symbol).
   */
  public DeclarationAssembler(
      DeclarationConsumer declarationConsumer, SeparatorFinder separatorFinder) {
    this.declarationConsumer = declarationConsumer;
    this.separatorFinder = separatorFinder;
  }

  /**
   * Should be called after all work for processing of individual buffer fragments is complete.
   *
   * @param fragments list of {@link ByteFragmentAtOffset} - pieces on the bounds of sub-fragments.
   * @throws GenericParsingException thrown by delegate {@link #declarationConsumer}
   */
  public void wrapUp(List<ByteFragmentAtOffset> fragments)
      throws GenericParsingException, IOException {
    fragments.sort(Comparator.comparingInt(ByteFragmentAtOffset::getFragmentOffset));

    List<ByteFragmentAtOffset> list = Lists.newArrayList();
    int previous = -1;
    for (ByteFragmentAtOffset edge : fragments) {
      int start = edge.getFragmentOffset();
      ByteBufferFragment fragment = edge.getFragment();
      if (previous >= 0 && previous != start) {
        sendMerged(list);
        list.clear();
      }
      list.add(edge);
      previous = start + fragment.length();
    }
    if (!list.isEmpty()) {
      sendMerged(list);
    }
  }

  private void sendMerged(List<ByteFragmentAtOffset> list)
      throws GenericParsingException, IOException {
    Preconditions.checkArgument(!list.isEmpty());
    ByteFragmentAtOffset first = list.get(0);
    if (list.size() == 1) {
      declarationConsumer.declaration(first);
      return;
    }

    // 1. We merge all the passed fragments into one fragment.
    // 2. We check 6 bytes at the connection of two fragments, 3 bytes in each part:
    // separator can consist of 4 bytes (<escape>/r/n<indent>),
    // so in case only a part of the separator is in one of the fragments,
    // we get 3 bytes in one part and one byte in the other.
    // 3. We record the ranges of at most 6 bytes at the connections of the fragments into
    // interestingRanges.
    // 4. Later we will check only interestingRanges for separators, and create corresponding
    // fragments; the underlying common ByteBuffer will be reused, so we are not performing
    // extensive copying.
    int firstOffset = first.getBufferOffset();
    List<ByteBufferFragment> fragments = new ArrayList<>();
    List<Range<Integer>> interestingRanges = Lists.newArrayList();
    int fragmentShift = 0;
    for (ByteFragmentAtOffset byteFragmentAtOffset : list) {
      ByteBufferFragment fragment = byteFragmentAtOffset.getFragment();
      fragments.add(fragment);
      if (fragmentShift > 0) {
        // We are only looking for the separators between fragments.
        int start = Math.max(0, fragmentShift - 3);
        int end = fragmentShift + Math.min(4, fragment.length());
        // Assert that the ranges are not intersecting, otherwise the code that iterates ranges
        // will work incorrectly.
        Preconditions.checkState(
            interestingRanges.isEmpty()
                || Iterables.getLast(interestingRanges).upperEndpoint() < start);
        interestingRanges.add(Range.openClosed(start, end));
      }
      fragmentShift += fragment.length();
    }

    ByteBufferFragment merged = ByteBufferFragment.merge(fragments);

    int newOffset;
    if (Iterables.getLast(list).getBufferOffset() == firstOffset) {
      // If all fragment offsets were the same (which is the case if all their originating buffers
      // were the same), then the merged fragment has the start index of the first originating
      // fragment, and the end index of the last originating fragment.
      // The old offset can be used without issue.
      newOffset = firstOffset;
    } else {
      // If fragment offsets differed, then the new merged fragment has a different offset.
      // In this case, the merged fragment has relative indices [0, len), which means that its
      // offset should reflect the absolute "real" start offset.
      newOffset = first.getFragmentOffset();
    }

    int previousEnd = 0;
    for (Range<Integer> range : interestingRanges) {
      int idx =
          separatorFinder.findNextSeparator(merged, range.lowerEndpoint(), range.upperEndpoint());
      if (idx >= 0) {
        // There should always be a previous fragment, as we are checking non-intersecting ranges,
        // starting from the connection point between first and second fragments.
        Preconditions.checkState(idx > previousEnd);
        declarationConsumer.declaration(
            new ByteFragmentAtOffset(newOffset, merged.subFragment(previousEnd, idx + 1)));
        previousEnd = idx + 1;
      }
    }
    declarationConsumer.declaration(
        new ByteFragmentAtOffset(newOffset, merged.subFragment(previousEnd, merged.length())));
  }
}
