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

import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;

/**
 * An abstraction for reading input from a file or taking it as a pre-cooked
 * char[] or String.
 */
public abstract class ParserInputSource {

  protected ParserInputSource() {}

  /**
   * Returns the content of the input source.
   */
  public abstract char [] getContent();

  /**
   * Returns the path of the input source. Note: Once constructed, this object
   * will never re-read the content from path.
   */
  public abstract PathFragment getPath();

  public static ParserInputSource create(byte[] bytes, PathFragment path) throws IOException {
    char[] content = convertFromLatin1(bytes);
    return create(content, path);
  }

  /**
   * Create an input source from the given content, and associate path with
   * this source.  Path will be used in error messages etc. but we will *never*
   * attempt to read the content from path.
   */
  public static ParserInputSource create(String content, PathFragment path) {
    return create(content.toCharArray(), path);
  }

  /**
   * Create an input source from the given content, and associate path with
   * this source.  Path will be used in error messages etc. but we will *never*
   * attempt to read the content from path.
   */
  public static ParserInputSource create(final char[] content, final PathFragment path) {
    return new ParserInputSource() {

      @Override
      public char[] getContent() {
        return content;
      }

      @Override
      public PathFragment getPath() {
        return path;
      }
    };
  }

  private static char[] convertFromLatin1(byte[] content) {
    char[] latin1 = new char[content.length];
    for (int i = 0; i < latin1.length; i++) { // yeah, latin1 is this easy! :-)
      latin1[i] = (char) (0xff & content[i]);
    }
    return latin1;
  }
}
