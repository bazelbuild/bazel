// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.includescanning;

import com.google.common.base.CharMatcher;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion.Kind;

/** Parses swig files and extracts their includes (%include / %extern / %import). */
class SwigIncludeParser extends IncludeParser {

  SwigIncludeParser() {
    // There are no preprocessor-macro hints for swig.
    super(/* hints= */ null);
  }

  private static int skipParentheses(byte[] chars, int pos, int end) {
    // TODO(bazel-team): In theory this could be multiline, but the include scanner currently works
    // on a single line.
    int openedParentheses = 1;
    if (pos >= end || chars[pos] != '(') {
      return pos;
    }
    pos++;
    while (openedParentheses > 0 && pos < end) {
      if (chars[pos] == '(') {
        openedParentheses++;
      } else if (chars[pos] == ')') {
        openedParentheses--;
      }
      pos++;
    }
    return pos;
  }

  /** See javadoc for {@link IncludeParser#getFileType()} */
  @Override
  protected GrepIncludesFileType getFileType() {
    return GrepIncludesFileType.SWIG;
  }

  @Override
  protected IncludesKeywordData expectIncludeKeyword(byte[] chars, int pos, int end) {
    int start = skipWhitespace(chars, pos, end);
    if ((pos = expect(chars, start, end, "%include")) == -1
        && (pos = expect(chars, start, end, "%extern")) == -1
        && (pos = expect(chars, start, end, "%import")) == -1) {
      return IncludesKeywordData.NONE;
    }
    int npos = skipWhitespace(chars, pos, end);
    npos = skipParentheses(chars, npos, end);
    npos = skipWhitespace(chars, npos, end);
    if (npos > pos) {
      return IncludesKeywordData.importOrSwig(npos);
    }
    return IncludesKeywordData.NONE;
  }

  @Override
  protected boolean isValidInclusionKind(Kind kind) {
    return !kind.isNext();
  }

  @Override
  protected Inclusion createOtherInclusion(String inclusionContent) {
    if (inclusionContent.startsWith("/")) {
      return null; // Ignore absolute path names.
    }

    // Truncate comments after filename.
    int index = inclusionContent.indexOf("//");
    if (index > 0) {
      inclusionContent = inclusionContent.substring(0, index);
    }
    index = inclusionContent.indexOf("/*");
    if (index > 0) {
      inclusionContent = inclusionContent.substring(0, index);
    }
    // Trim whitespace.
    inclusionContent = CharMatcher.whitespace().trimFrom(inclusionContent);

    // Treat swig inclusions w/o quotes or angle brackets as quoted inclusions.
    return inclusionContent.length() > 0 ? Inclusion.create(inclusionContent, Kind.QUOTE) : null;
  }
}
