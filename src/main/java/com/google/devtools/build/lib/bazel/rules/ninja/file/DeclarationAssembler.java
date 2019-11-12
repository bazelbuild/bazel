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
import java.util.Comparator;
import java.util.List;

/**
 * A {@link BufferSplitter} callback interface implementation, that assembles fragments of
 * declarations (that may occur on the edges of byte buffer fragments) together and passes all
 * declarations to delegate {@link DeclarationConsumer}, which does further processing / parsing.
 */
public class DeclarationAssembler {
  private final DeclarationConsumer declarationConsumer;
  private final SeparatorPredicate separatorPredicate;

  /**
   * @param declarationConsumer delegate declaration consumer for actual processing / parsing
   * @param separatorPredicate predicate used to determine if two fragments should be separate
   *     declarations (in the Ninja case, if the new line starts with a space, it should be treated
   *     as a part of the previous declaration, i.e. the separator is longer then one symbol).
   */
  public DeclarationAssembler(
      DeclarationConsumer declarationConsumer, SeparatorPredicate separatorPredicate) {
    this.declarationConsumer = declarationConsumer;
    this.separatorPredicate = separatorPredicate;
  }

  /**
   * Should be called after all work for processing of individual buffer fragments is complete.
   *
   * @param fragments list of {@link ByteFragmentAtOffset} - pieces on the bounds of sub-fragments.
   * @throws GenericParsingException thrown by delegate {@link #declarationConsumer}
   */
  public void wrapUp(List<ByteFragmentAtOffset> fragments) throws GenericParsingException {
    fragments.sort(Comparator.comparingInt(ByteFragmentAtOffset::getRealStartOffset));

    List<ByteFragmentAtOffset> list = Lists.newArrayList();
    int previous = -1;
    for (ByteFragmentAtOffset edge : fragments) {
      int start = edge.getRealStartOffset();
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

  private void sendMerged(List<ByteFragmentAtOffset> list) throws GenericParsingException {
    int offset = -1;
    List<ByteBufferFragment> leftPart = Lists.newArrayList();

    for (ByteFragmentAtOffset edge : list) {
      ByteBufferFragment sequence = edge.getFragment();
      // If the new sequence is separate from already collected parts,
      // merge them and feed to consumer.
      if (!leftPart.isEmpty()) {
        ByteBufferFragment lastPart = Iterables.getLast(leftPart);
        // The order of symbols: previousInOld, lastInOld, currentInNew, nextInNew.
        byte previousInOld = lastPart.length() == 1 ? 0 : lastPart.byteAt(lastPart.length() - 2);
        byte lastInOld = lastPart.byteAt(lastPart.length() - 1);
        byte currentInNew = sequence.byteAt(0);
        byte nextInNew = sequence.length() == 1 ? 0 : sequence.byteAt(1);

        // <symbol> | \n<non-space>
        if (separatorPredicate.test(lastInOld, currentInNew, nextInNew)) {
          // Add separator to the end of the accumulated sequence
          leftPart.add(sequence.subFragment(0, 1));
          ByteFragmentAtOffset byteFragmentAtOffset =
              new ByteFragmentAtOffset(edge.getOffset(), ByteBufferFragment.merge(leftPart));
          declarationConsumer.declaration(byteFragmentAtOffset);
          leftPart.clear();
          // Cutting out the separator in the beginning
          if (sequence.length() > 1) {
            leftPart.add(sequence.subFragment(1, sequence.length()));
            offset = edge.getOffset();
          }
          continue;
        }

        // <symbol>\n | <non-space>
        if (separatorPredicate.test(previousInOld, lastInOld, currentInNew)) {
          ByteFragmentAtOffset byteFragmentAtOffset =
              new ByteFragmentAtOffset(edge.getOffset(), ByteBufferFragment.merge(leftPart));
          declarationConsumer.declaration(byteFragmentAtOffset);
          leftPart.clear();
        }
      }

      leftPart.add(sequence);
      if (offset == -1) {
        offset = edge.getOffset();
      }
    }
    if (!leftPart.isEmpty()) {
      Preconditions.checkState(offset >= 0);
      ByteFragmentAtOffset byteFragmentAtOffset =
          new ByteFragmentAtOffset(offset, ByteBufferFragment.merge(leftPart));
      declarationConsumer.declaration(byteFragmentAtOffset);
    }
  }
}
