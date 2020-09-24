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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.protobuf.TextFormat;
import java.io.BufferedOutputStream;

/**
 * A simple {@link BuildEventTransport} that writes the text representation of the protocol-buffer
 * representation of the events to a file.
 *
 * <p>This class is used for debugging.
 */
public final class TextFormatFileTransport extends FileTransport {
  public TextFormatFileTransport(
      BufferedOutputStream outputStream,
      BuildEventProtocolOptions options,
      BuildEventArtifactUploader uploader,
      ArtifactGroupNamer namer) {
    super(outputStream, options, uploader, namer);
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }

  @Override
  protected byte[] serializeEvent(BuildEventStreamProtos.BuildEvent buildEvent) {
    String protoTextRepresentation = TextFormat.printToString(buildEvent);
    return ("event {\n" + protoTextRepresentation + "}\n\n").getBytes(UTF_8);
  }
}
