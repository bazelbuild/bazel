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

package com.google.devtools.build.lib.shell;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Provides sinks for input streams.  Continuously read an input stream
 * until the end-of-file is encountered.  The stream may be redirected to
 * an {@link OutputStream}, or discarded.
 * <p>
 * This class is useful for handing the {@code stdout} and {@code stderr}
 * streams from a {@link Process} started with {@link Runtime#exec(String)}.
 * If these streams are not consumed, the Process may block resulting in a
 * deadlock.
 *
 * @see <a href="http://www.javaworld.com/javaworld/jw-12-2000/jw-1229-traps.html">
 *      JavaWorld: When Runtime.exec() won&apos;t</a>
 */
public final class InputStreamSink {

  /**
   * Black hole into which bytes are sometimes discarded by {@link NullSink}.
   * It is shared by all threads since the actual contents of the buffer
   * are irrelevant.
   */
  private static final byte[] DISCARD = new byte[4096];

  // Suppresses default constructor; ensures non-instantiability
  private InputStreamSink() {}

  /**
   * A {@link Thread} which reads and discards data from an
   * {@link InputStream}.
   */
  private static class NullSink implements Runnable {
    private final InputStream in;

    public NullSink(InputStream in) {
      this.in = in;
    }

    @Override
    public void run() {
      try {
        try {
          // Attempt to just skip all input
          do {
            in.skip(Integer.MAX_VALUE);
          } while (in.read() != -1); // Need to test for EOF
        } catch (IOException ioe) {
          // Some streams throw IOException when skip() is called;
          // resort to reading off all input with read():
          while (in.read(DISCARD) != -1) {
            // no loop body
          }
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * A {@link Thread} which reads data from an {@link InputStream},
   * and translates it into an {@link OutputStream}.
   */
  private static class CopySink implements Runnable {

    private final InputStream in;
    private final OutputStream out;

    public CopySink(InputStream in, OutputStream out) {
      this.in = in;
      this.out = out;
    }

    @Override
    public void run() {
      try {
        byte[] buffer = new byte[2048];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) >= 0) {
          out.write(buffer, 0, bytesRead);
          out.flush();
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * Creates a {@link Runnable} which consumes the provided
   * {@link InputStream} 'in', discarding its contents.
   */
  public static Runnable newRunnableSink(InputStream in) {
    if (in == null) {
      throw new NullPointerException("in");
    }
    return new NullSink(in);
  }

  /**
   * Creates a {@link Runnable} which copies everything from 'in'
   * to 'out'. 'out' will be written to and flushed after each
   * read from 'in'. However, 'out' will not be closed.
   */
  public static Runnable newRunnableSink(InputStream in, OutputStream out) {
    if (in == null) {
      throw new NullPointerException("in");
    }
    if (out == null) {
      throw new NullPointerException("out");
    }
    return new CopySink(in, out);
  }
}
