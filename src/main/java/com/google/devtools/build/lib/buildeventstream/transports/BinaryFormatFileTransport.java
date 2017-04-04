// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * A simple {@link BuildEventTransport} that writes varint delimited binary representation of event
 * {@link BuildEvent} protocol-buffers to a file. Files written by this Transport can be read by
 * successive calls of {code BuildEvent.Builder#mergeDelimitedFrom(InputStream)} (or the static
 * method {@code BuildEvent.parseDelimitedFrom(InputStream)}).
 */
public final class BinaryFormatFileTransport implements BuildEventTransport {
  private final BufferedOutputStream out;
  private final PathConverter pathConverter;

  public BinaryFormatFileTransport(String path, PathConverter pathConverter)
      throws IOException {
    this.out = new BufferedOutputStream(new FileOutputStream(new File(path)));
    this.pathConverter = pathConverter;
  }

  @Override
  public synchronized void sendBuildEvent(BuildEvent event) throws IOException {
    BuildEventConverters converters =
        new BuildEventConverters() {
          @Override
          public PathConverter pathConverter() {
            return pathConverter;
          }
        };
    event.asStreamProto(converters).writeDelimitedTo(out);
    out.flush();
  }

  @Override
  public void close() throws IOException {
    out.close();
  }
}
