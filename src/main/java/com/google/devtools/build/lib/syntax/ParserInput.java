// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** The apparent name and contents of a source file, for consumption by the parser. */
public final class ParserInput {

  private final char[] content;
  private final PathFragment path;

  private ParserInput(char[] content, @Nullable PathFragment path) {
    this.content = content;
    this.path = path == null ? PathFragment.EMPTY_FRAGMENT : path;
  }

  /** Returns the content of the input source. Callers must not modify the result. */
  char[] getContent() {
    return content;
  }

  /**
   * Returns the (non-null) apparent file name of the input source, for use in error messages; the
   * file will not be opened.
   */
  // TODO(adonovan): use Strings, to avoid dependency on vfs; but first we need to avoid depending
  // on events.Location.
  public PathFragment getPath() {
    return path;
  }

  /** Returns an unnamed input source that reads from a list of strings, joined by newlines. */
  public static ParserInput fromLines(String... lines) {
    return create(Joiner.on("\n").join(lines), null);
  }

  /**
   * Returns an import source that reads from a Latin-1 encoded byte array. The path specifies the
   * name of the file, for use in source locations and error messages; a null path implies the empty
   * string.
   */
  public static ParserInput create(byte[] bytes, @Nullable PathFragment path) {
    char[] content = convertFromLatin1(bytes);
    return new ParserInput(content, path);
  }

  /**
   * Create an input source from the given content, and associate path with this source. Path will
   * be used in error messages etc. but we will *never* attempt to read the content from path. A
   * null path implies the empty string.
   */
  public static ParserInput create(String content, @Nullable PathFragment path) {
    return create(content.toCharArray(), path);
  }

  /**
   * Create an input source from the given content, and associate path with this source. Path will
   * be used in error messages etc. but we will *never* attempt to read the content from path.
   */
  public static ParserInput create(char[] content, PathFragment path) {
    return new ParserInput(content, path);
  }

  private static char[] convertFromLatin1(byte[] content) {
    char[] latin1 = new char[content.length];
    for (int i = 0; i < latin1.length; i++) { // yeah, latin1 is this easy! :-)
      latin1[i] = (char) (0xff & content[i]);
    }
    return latin1;
  }
}
