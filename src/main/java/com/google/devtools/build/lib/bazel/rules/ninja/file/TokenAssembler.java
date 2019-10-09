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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

/**
 * A {@link BufferTokenizer} callback interface implementation, that assembles fragments of
 * tokens (that may occur on the edges of char[] buffer fragments) together and passes all tokens to
 * delegate {@link TokenConsumer}, which does further processing / parsing.
 */
public class TokenAssembler implements TokenAndFragmentsConsumer {
  private final Map<Integer, ArrayViewCharSequence> fragments;
  private final TokenConsumer tokenConsumer;
  private final SeparatorPredicate separatorPredicate;
  private final List<Throwable> throwables;

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
    fragments = Collections.synchronizedNavigableMap(Maps.newTreeMap());
    throwables = Lists.newCopyOnWriteArrayList();
  }

  @Override
  public void token(CharSequence sequence) throws GenericParsingException {
    tokenConsumer.token(sequence);
  }

  @Override
  public void fragment(int offset, ArrayViewCharSequence sequence) {
    fragments.put(offset + sequence.getStartIncl(), sequence);
  }

  @Override
  public void error(Throwable throwable) {
    throwables.add(throwable);
  }

  /**
   * Should be called after all work for processing of individual buffer fragments is complete.
   * @throws GenericParsingException thrown by delegate {@link #tokenConsumer}
   */
  public void wrapUp() throws GenericParsingException {
    if (!throwables.isEmpty()) {
      throw new RuntimeException(Iterables.getFirst(throwables, null));
    }
    List<ArrayViewCharSequence> list = Lists.newArrayList();
    int previous = -1;
    for (Map.Entry<Integer, ArrayViewCharSequence> entry : fragments.entrySet()) {
      if (previous >= 0 && previous != entry.getKey()) {
        sendMerged(list);
        list.clear();
      }
      list.add(entry.getValue());
      previous = (entry.getKey() - entry.getValue().getStartIncl()) + entry.getValue().getEndExcl();
    }
    if (! list.isEmpty()) {
      sendMerged(list);
    }
  }

  private void sendMerged(List<ArrayViewCharSequence> list) throws GenericParsingException {
    List<ArrayViewCharSequence> leftPart = Lists.newArrayList();

    for (ArrayViewCharSequence sequence : list) {
      if (!leftPart.isEmpty()) {
        ListIterator<Character> leftIterator = leftPart.get(leftPart.size() - 1).iteratorAtEnd();
        if (separatorPredicate.splitAdjacent(leftIterator, sequence.iterator())) {
          tokenConsumer.token(ArrayViewCharSequence.merge(leftPart));
          leftPart.clear();
        }
      }
      leftPart.add(sequence);
    }
    if (!leftPart.isEmpty()) {
      tokenConsumer.token(ArrayViewCharSequence.merge(leftPart));
    }
  }
}
