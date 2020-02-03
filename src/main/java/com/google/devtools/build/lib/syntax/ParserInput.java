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
import com.google.common.base.Preconditions;

/**
 * The apparent name and contents of a source file, for consumption by the parser. The file name
 * appears in the location information in the syntax tree, and in error messages, but the Starlark
 * interpreter will not attempt to open the file.
 */
public final class ParserInput {

  private final String file;
  private final char[] content;

  private ParserInput(char[] content, String file) {
    this.content = content;
    this.file = Preconditions.checkNotNull(file);
  }

  /** Returns the content of the input source. Callers must not modify the result. */
  char[] getContent() {
    return content;
  }

  /** Returns the apparent file name of the input source. */
  public String getFile() {
    return file;
  }

  /** Returns an unnamed input source that reads from a list of strings, joined by newlines. */
  public static ParserInput fromLines(String... lines) {
    return create(Joiner.on("\n").join(lines), "");
  }

  /** Returns an import source that reads from a Latin-1 encoded byte array. */
  public static ParserInput create(byte[] bytes, String file) {
    char[] content = convertFromLatin1(bytes);
    return new ParserInput(content, file);
  }

  /** Returns an input source that reads from the given string. */
  public static ParserInput create(String content, String file) {
    return create(content.toCharArray(), file);
  }

  /** Returns an input source that reads from the given char array. */
  public static ParserInput create(char[] content, String file) {
    return new ParserInput(content, file);
  }

  private static char[] convertFromLatin1(byte[] content) {
    char[] latin1 = new char[content.length];
    for (int i = 0; i < latin1.length; i++) { // yeah, latin1 is this easy! :-)
      latin1[i] = (char) (0xff & content[i]);
    }
    return latin1;
  }
}
