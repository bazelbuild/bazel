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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion.Kind;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/**
 * Helper class that deals with the processing of a given file's inclusions. Extracted for use in
 * multiple include scanning implementations.
 */
class IncludeScannerHelper {
  private final List<PathFragment> includePaths;
  private final List<PathFragment> quoteIncludePaths;
  private final Artifact source;

  private boolean hasQuoteContextPathPos = false;
  private int quoteContextPathPos = -1;
  private  boolean hasAngleContextPathPos = false;
  private int angleContextPathPos = -1;

  IncludeScannerHelper(List<PathFragment> includePaths, List<PathFragment> quoteIncludePaths,
      Artifact source) {
    this.includePaths = includePaths;
    this.quoteIncludePaths = quoteIncludePaths;
    this.source = source;
  }

  /**
   * Finds {@code source} in the given {@includePaths} as if it were #included as {@include}.
   *
   * @return the index in {@includePaths} a #include_next directive should start searching from,
   *         or -1 if the source file was not found.
   */
  private static int findContextPathPos(PathFragment include, Artifact source,
      List<PathFragment> includePaths) {
    for (int i = 0; i < includePaths.size(); ++i) {
      PathFragment execPath = includePaths.get(i);
      if (execPath.getRelative(include).equals(source.getExecPath())) {
        return i + 1;
      }
    }
    return -1;
  }

  /**
   * Most inclusions become inclusions with context as-is -- the context of their including file is
   * their context. However, in the case of a top-level #include_next inclusion, a more involved
   * procedure is needed to find the proper context.
   */
  InclusionWithContext createInclusionWithContext(Inclusion inclusion, int contextPathPos,
      Kind contextKind) {
    if (inclusion.kind.isNext() && contextPathPos == -1) {
      // Special handling for the case when a #include_next directive is found without context,
      // e. g., in the main source file. This is the case if either:
      // 1. We do header preprocessing / parsing to check strict layering, or when building
      //    header modules; in this case we emulate clang's behavior; gcc's implementation
      //    diverges from clang here, but gcc does not support these use cases.
      // 2. A .c / .cc file contains #include_next. Here the behavior of clang/gcc is different
      //    from what we implement, but it will lead to an error anyway.

      // First, try to find the source of the inclusion via a header search.
      if (inclusion.kind == Kind.NEXT_QUOTE) {
        if (!hasQuoteContextPathPos) {
          quoteContextPathPos =
              findContextPathPos(inclusion.pathFragment, source, quoteIncludePaths);
          hasQuoteContextPathPos = true;
        }
        contextPathPos = quoteContextPathPos;
      } else {
        if (!hasAngleContextPathPos) {
          angleContextPathPos =
              findContextPathPos(inclusion.pathFragment, source, includePaths);
          hasAngleContextPathPos = true;
        }
        contextPathPos = angleContextPathPos;
      }
      Kind kind = (inclusion.kind == Kind.NEXT_QUOTE) ? Kind.QUOTE : Kind.ANGLE;
      if (contextPathPos != -1) {
        // If the source was found via the header search, assume that is a valid context to
        // start the search from.
        // The context kind was previously unknown, as this inclusion came from a top-level
        // source; assume the context kind is the same as the kind of the include_next
        // directive.
        contextKind = kind;
      } else {
        // If the source was not found, #include_next behaves exactly like #include.
        inclusion = Inclusion.create(inclusion.pathFragment, kind);
      }
    }
    return new InclusionWithContext(inclusion, contextPathPos, contextKind);
  }
}
