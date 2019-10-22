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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteBufferFragment;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.file.NinjaSeparatorPredicate;
import java.nio.charset.Charset;
import java.util.List;

public class NinjaLineSplitterUtil {
  public static List<String> splitIntoLines(ByteBufferFragment fragment, Charset charset) {
    String sequence = fragment.toString(charset);
    List<String> result = Lists.newArrayList();
    boolean skipComment = false;
    int start = 0;
    for (int i = 0; i < sequence.length(); i++) {
      char ch = sequence.charAt(i);
      if ('\n' == ch || '#' == ch) {
        if (!skipComment && i > start) {
          result.add(sequence.substring(start, i));
        }
        start = i + 1;
        skipComment = '#' == ch;
      }
    }
    if (start < sequence.length()) {
      result.add(sequence.substring(start));
    }
    return result;
  }

  public static byte[] getFirstWordFragment(ByteBufferFragment fragment)
      throws GenericParsingException {
    Preconditions.checkState(fragment.length() > 0);
    int index = 0;

    if (index < fragment.length()
        && !NinjaSeparatorPredicate.isNotSpace(fragment.byteAt(index))) {
      throw new GenericParsingException(
          String.format("Expected the line to not start with spaces: '%s'", fragment));
    }
    ++ index;
    while (index < fragment.length()
        && NinjaSeparatorPredicate.isNotSpace(fragment.byteAt(index))) {
      ++ index;
    }
    byte[] result = new byte[index];
    fragment.getBytes(result, 0, index);
    return result;
  }
}
