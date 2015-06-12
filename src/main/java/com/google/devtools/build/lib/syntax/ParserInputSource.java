// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;

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

  /**
   * Create an input source instance by (eagerly) reading from the file at
   * path. The file is assumed to be ISO-8859-1 encoded and smaller than
   * 2 Gigs - these assumptions are reasonable for BUILD files, which is
   * all we care about here.
   */
  public static ParserInputSource create(Path path) throws IOException {
    char[] content = FileSystemUtils.readContentAsLatin1(path);
    if (path.getFileSize() > content.length) {
      // This assertion is to help diagnose problems arising from the
      // filesystem;  see bugs and #859334 and #920195.
      throw new IOException("Unexpected short read from file '" + path
          + "' (expected " + path.getFileSize() + ", got " + content.length + " bytes)");
    }
    return create(content, path.asFragment());
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

  /**
   * Create an input source from the given input stream, and associate path
   * with this source.  'path' will be used in error messages, etc, but will
   * not (in general) be used to to read the content from path.
   *
   * <p>(The exception is the case in which Python pre-processing is required; the
   * path will be used to provide the input to the Python pre-processor.
   * Arguably, we should just send the content as input to the subprocess
   * instead of using the path, but it's not clear it's worth the effort.)
   */
  public static ParserInputSource create(InputStream in, Path path) throws IOException {
    try {
      return create(new String(FileSystemUtils.readContentAsLatin1(in)), path.asFragment());
    } finally {
      in.close();
    }
  }
}
