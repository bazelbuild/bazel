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
import com.google.protobuf.TextFormat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * A simple {@link BuildEventTransport} that writes the text representation of the protocol-buffer
 * representation of the events to a file. This is mainly useful for debugging.
 */
public final class TextFormatFileTransport implements BuildEventTransport {
  private FileOutputStream out;
  private final PathConverter pathConverter;

  public TextFormatFileTransport(String path, PathConverter pathConverter)
      throws IOException {
    this.out = new FileOutputStream(new File(path));
    this.pathConverter = pathConverter;
  }

  @Override
  public synchronized void sendBuildEvent(BuildEvent event) throws IOException {
    if (out != null) {
      BuildEventConverters converters =
          new BuildEventConverters() {
            @Override
            public PathConverter pathConverter() {
              return pathConverter;
            }
          };
      String protoTextRepresentation = TextFormat.printToString(event.asStreamProto(converters));
      out.write(("event {\n" + protoTextRepresentation + "}\n\n").getBytes(StandardCharsets.UTF_8));
      out.flush();
    }
  }

  @Override
  public void close() throws IOException {
    if (out != null) {
      out.close();
    }
  }
}
