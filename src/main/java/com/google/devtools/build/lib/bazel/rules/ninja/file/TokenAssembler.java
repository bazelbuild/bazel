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
import com.google.devtools.build.lib.util.Pair;
import java.util.Comparator;
import java.util.List;

/**
 * A {@link BufferTokenizer} callback interface implementation, that assembles fragments of
 * tokens (that may occur on the edges of byte buffer fragments) together and passes all tokens to
 * delegate {@link TokenConsumer}, which does further processing / parsing.
 */
public class TokenAssembler {
  private final TokenConsumer tokenConsumer;
  private final SeparatorPredicate separatorPredicate;

  /**
   * @param tokenConsumer delegate token consumer for actual processing / parsing
   * @param separatorPredicate predicate used to determine if two fragments should be separate
   * tokens (in the Ninja case, if the new line starts with a space, it should be treated as a part
   * of the previous token, i.e. the separator is longer then one symbol).
   */
  public TokenAssembler(
      TokenConsumer tokenConsumer,
      SeparatorPredicate separatorPredicate) {
    this.tokenConsumer = tokenConsumer;
    this.separatorPredicate = separatorPredicate;
  }

  /**
   * Should be called after all work for processing of individual buffer fragments is complete.
   * @param fragments list of pairs (offset from the beginning of file, buffer fragment) of
   * the pieces on the bounds of sub-fragments.
   * @throws GenericParsingException thrown by delegate {@link #tokenConsumer}
   */
  public void wrapUp(List<Pair<Integer, ByteBufferFragment>> fragments)
      throws GenericParsingException {
    fragments.sort(Comparator.comparingInt(Pair::getFirst));

    List<ByteBufferFragment> list = Lists.newArrayList();
    int previous = -1;
    for (Pair<Integer, ByteBufferFragment> pair : fragments) {
      int start = Preconditions.checkNotNull(pair.getFirst());
      ByteBufferFragment fragment = Preconditions.checkNotNull(pair.getSecond());
      if (previous >= 0 && previous != start) {
        sendMerged(list);
        list.clear();
      }
      list.add(fragment);
      previous = start + fragment.length();
    }
    if (! list.isEmpty()) {
      sendMerged(list);
    }
  }

  private void sendMerged(List<ByteBufferFragment> list) throws GenericParsingException {
    List<ByteBufferFragment> leftPart = Lists.newArrayList();

    for (ByteBufferFragment sequence : list) {
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
          tokenConsumer.token(ByteBufferFragment.merge(leftPart));
          leftPart.clear();
          // Cutting out the separator in the beginning
          if (sequence.length() > 1) {
            leftPart.add(sequence.subFragment(1, sequence.length()));
          }
          continue;
        }

        // <symbol>\n | <non-space>
        if (separatorPredicate.test(previousInOld, lastInOld, currentInNew)) {
          tokenConsumer.token(ByteBufferFragment.merge(leftPart));
          leftPart.clear();
        }
      }

      leftPart.add(sequence);
    }
    if (!leftPart.isEmpty()) {
      tokenConsumer.token(ByteBufferFragment.merge(leftPart));
    }
  }
}
