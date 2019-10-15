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
import java.util.function.BiPredicate;

/**
 * A {@link BufferTokenizer} callback interface implementation, that assembles fragments of
 * tokens (that may occur on the edges of byte buffer fragments) together and passes all tokens to
 * delegate {@link TokenConsumer}, which does further processing / parsing.
 */
public class TokenAssembler {
  private final TokenConsumer tokenConsumer;
  private final BiPredicate<Byte, Byte> separatorPredicate;

  /**
   * @param tokenConsumer delegate token consumer for actual processing / parsing
   * @param separatorPredicate predicate used to determine if two fragments should be separate
   * tokens (in the Ninja case, if the new line starts with a space, it should be treated as a part
   * of the previous token, i.e. the separator is longer then one symbol).
   */
  public TokenAssembler(
      TokenConsumer tokenConsumer,
      BiPredicate<Byte, Byte> separatorPredicate) {
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
      if (!leftPart.isEmpty()) {
        ByteBufferFragment lastPart = Iterables.getLast(leftPart);
        boolean isSeparator = separatorPredicate
            .test(lastPart.byteAt(lastPart.length() - 1), sequence.byteAt(0));
        // If the new sequence is separate from already collected parts,
        // merge them and feed to consumer.
        if (isSeparator) {
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
