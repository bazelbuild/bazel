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
import java.nio.ByteBuffer;
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
   * Reuse this buffer fragment, wrapping {@link #edgeBytes},
   * for passing it to {@link SeparatorFinder}.
   */
  private final ByteBufferFragment edgeFragment;
  private final byte[] edgeBytes;

  /**
   * @param declarationConsumer delegate declaration consumer for actual processing / parsing
   * @param separatorFinder callback used to determine if two fragments should be separate
   * declarations (in the Ninja case, if the new line starts with a space, it should be treated
   * as a part of the previous declaration, i.e. the separator is longer then one symbol).
   */
  public DeclarationAssembler(
      DeclarationConsumer declarationConsumer, SeparatorFinder separatorFinder) {
    this.declarationConsumer = declarationConsumer;
    this.separatorFinder = separatorFinder;
    edgeBytes = new byte[6];
    edgeFragment = new ByteBufferFragment(ByteBuffer.wrap(edgeBytes), 0, 6);
  }

  /**
   * Should be called after all work for processing of individual buffer fragments is complete.
   *
   * @param fragments list of {@link BufferEdge} - pieces on the bounds of sub-fragments.
   * @throws GenericParsingException thrown by delegate {@link #declarationConsumer}
   */
  public void wrapUp(List<BufferEdge> fragments) throws GenericParsingException {
    fragments.sort(Comparator.comparingInt(BufferEdge::getRealStartOffset));

    List<ByteBufferFragment> list = Lists.newArrayList();
    int previous = -1;
    for (BufferEdge edge : fragments) {
      int start = edge.getRealStartOffset();
      ByteBufferFragment fragment = edge.getFragment();
      if (previous >= 0 && previous != start) {
        sendMerged(list);
        list.clear();
      }
      list.add(fragment);
      previous = start + fragment.length();
    }
    if (!list.isEmpty()) {
      sendMerged(list);
    }
  }

  private void sendMerged(List<ByteBufferFragment> list) throws GenericParsingException {
    if (list.size() == 1) {
      declarationConsumer.declaration(list.get(0));
      return;
    }

    List<ByteBufferFragment> leftPart = Lists.newArrayList();

    for (ByteBufferFragment sequence : list) {
      // If the new sequence is separate from already collected parts,
      // merge them and feed to consumer.
      if (!leftPart.isEmpty()) {
        ByteBufferFragment lastPart = Iterables.getLast(leftPart);

        final int offset = copyToEdgeBytes(lastPart, sequence);
        int lastPartLength = Math.min(3, lastPart.length());
        int nextSeparator = separatorFinder.findNextSeparator(edgeFragment, offset) - offset;
        if (nextSeparator > 0) {
          if (nextSeparator >= lastPartLength) {
            // Add separator to the end of the accumulated sequence
            int endSeparator = nextSeparator - lastPartLength + 1;
            leftPart.add(sequence.subFragment(0, endSeparator));
            declarationConsumer.declaration(ByteBufferFragment.merge(leftPart));
            leftPart.clear();
            // Cutting out the separator in the beginning
            if (sequence.length() > endSeparator) {
              leftPart.add(sequence.subFragment(endSeparator, sequence.length()));
            }
          } else {
            declarationConsumer.declaration(ByteBufferFragment.merge(leftPart));
            leftPart.clear();
            leftPart.add(sequence);
          }
          continue;
        }
      }

      leftPart.add(sequence);
    }
    if (!leftPart.isEmpty()) {
      declarationConsumer.declaration(ByteBufferFragment.merge(leftPart));
    }
  }

  /**
   * Copy the end of the lastPart and the beginning of nextPart into the reused buffer edgeBytes:
   * [<at most 3 bytes of lastPart><at most 3 bytes of nextPart>]
   */
  private int copyToEdgeBytes(ByteBufferFragment lastPart, ByteBufferFragment nextPart) {
    int startLast = Math.max(0, lastPart.length() - 3);
    int endLast = lastPart.length();
    // Start in sequence is always 0.
    int endSequence = Math.min(3, nextPart.length());
    int lastPartLength = Math.min(3, lastPart.length());

    // Copy to the end of reused buffer, and pass offset to the resulting sequence
    // into findNextSeparator().
    final int offset = 6 - lastPartLength - endSequence;
    int idx = offset;
    for (int i = startLast; i < endLast; i++) {
      edgeBytes[idx ++] = lastPart.byteAt(i);
    }
    for (int i = 0; i < endSequence; i++) {
      edgeBytes[idx ++] = nextPart.byteAt(i);
    }
    return offset;
  }
}
