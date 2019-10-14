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
import java.util.List;
import java.util.ListIterator;

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
   * @throws GenericParsingException thrown by delegate {@link #tokenConsumer}
   */
  public void wrapUp(List<Integer> offsets, List<List<ByteBufferFragment>> fragments)
      throws GenericParsingException {
    Preconditions.checkState(offsets.size() == fragments.size());

    List<ByteBufferFragment> list = Lists.newArrayList();
    int previous = -1;
    for (int i = 0; i < offsets.size(); i++) {
      int offset = offsets.get(i);
      for (ByteBufferFragment fragment : fragments.get(i)) {
        int key = offset + fragment.getStartIncl();
        if (previous >= 0 && previous != key) {
          sendMerged(list);
          list.clear();
        }
        list.add(fragment);
        previous = offset + fragment.getEndExcl();
      }
    }
    if (! list.isEmpty()) {
      sendMerged(list);
    }
  }

  private void sendMerged(List<ByteBufferFragment> list) throws GenericParsingException {
    List<ByteBufferFragment> leftPart = Lists.newArrayList();

    for (ByteBufferFragment sequence : list) {
      if (!leftPart.isEmpty()) {
        ListIterator<Byte> leftIterator = Iterables.getLast(leftPart).iteratorAtEnd();
        if (separatorPredicate.splitAdjacent(leftIterator, sequence.iterator())) {
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
