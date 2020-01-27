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

package com.google.devtools.build.lib.util.io;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.io.OutputStream;

/**
 * A stream that writes to another one, emittig a prefix before every line
 * it emits. This stream will also add a newline for every flush; so it's not
 * useful for anything other than simple text data (e.g. log files). Here's
 * an example which demonstrates how an explicit flush or a flush caused by
 * a full buffer causes a newline to be added to the output.
 *
 * <code>
 * foo bar
 * baz ba[flush]ng
 * boo
 * </code>
 *
 * This results in this output being emitted:
 *
 * <code>
 * my prefix: foo bar
 * my prefix: ba
 * my prefix: ng
 * my prefix: boo
 * </code>
 */
public final class LinePrefixingOutputStream extends LineFlushingOutputStream {

  private byte[] linePrefix;
  private final OutputStream sink;

  public LinePrefixingOutputStream(String linePrefix, OutputStream sink) {
    this.linePrefix = linePrefix.getBytes(UTF_8);
    this.sink = sink;
  }

  @Override
  protected void flushingHook() throws IOException {
    synchronized (sink) {
      if (len == 0) {
        sink.flush();
        return;
      }
      byte lastByte = buffer[len - 1];
      boolean lineIsIncomplete = lastByte != NEWLINE;
      sink.write(linePrefix);
      sink.write(buffer, 0, len);
      if (lineIsIncomplete) {
        sink.write(NEWLINE);
      }
      sink.flush();
      len = 0;
    }
  }

}
