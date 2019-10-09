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

/**
 * Task for tokenizing the contents of the char[] buffer.
 * Intended to be called in parallel for the fragments of the char[] buffer for lexing the
 * contents into independent logical tokens.
 *
 * {@link ParallelFileProcessing}
 */
public class BufferTokenizer implements Runnable {
  private final char[] buffer;
  private final TokenAndFragmentsConsumer consumer;
  private final SeparatorPredicate separatorPredicate;
  private final int offset;
  private final int startIncl;
  private final int endExcl;

  /**
   * @param buffer buffer, fragment of which should be tokenized
   * @param consumer token consumer
   * @param separatorPredicate predicate for separating tokens
   * @param offset offset of this buffer (sum of the number of characters in all previous buffers,
   * read from this file)
   * @param startIncl start index of the buffer fragment, inclusive
   * @param endExcl end index of the buffer fragment, exclusive, or the size of the buffer,
   * if last symbol is included
   */
  public BufferTokenizer(char[] buffer,
      TokenAndFragmentsConsumer consumer,
      SeparatorPredicate separatorPredicate,
      int offset, int startIncl, int endExcl) {
    this.buffer = buffer;
    this.consumer = consumer;
    this.separatorPredicate = separatorPredicate;
    this.offset = offset;
    this.startIncl = startIncl;
    this.endExcl = endExcl;
  }

  @Override
  public void run() {
    try {
      int start = startIncl;
      CharacterArrayIterator iterator = new CharacterArrayIterator(buffer, startIncl, endExcl);
      while (iterator.hasNext()) {
        // isSeparator() returns true/false or null - in the case when only the start of the
        // separator is present in the end of the buffer fragment.
        Boolean isSeparator = separatorPredicate.isSeparator(iterator);
        if (Boolean.FALSE.equals(isSeparator)) {
          continue;
        }
        ArrayViewCharSequence fragment =
            new ArrayViewCharSequence(buffer, start, startIncl + iterator.nextIndex());
        if (Boolean.TRUE.equals(isSeparator) && start > startIncl) {
          consumer.token(fragment);
        } else {
          consumer.fragment(offset, fragment);
        }
        start = startIncl + iterator.nextIndex();
      }
      if (start < endExcl) {
        consumer.fragment(offset, new ArrayViewCharSequence(buffer, start, endExcl));
      }
    } catch (Throwable e){
      consumer.error(e);
    }
  }

}
