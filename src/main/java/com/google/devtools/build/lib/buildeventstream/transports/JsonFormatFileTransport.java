// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import java.io.BufferedOutputStream;

/**
 * A simple {@link BuildEventTransport} that writes the JSON representation of the protocol-buffer
 * representation of the events to a file.
 */
public final class JsonFormatFileTransport extends FileTransport {
  public JsonFormatFileTransport(
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
    String protoJsonRepresentation;
    try {
      protoJsonRepresentation =
          JsonFormat.printer().omittingInsignificantWhitespace().print(buildEvent) + "\n";
    } catch (InvalidProtocolBufferException e) {
      // We don't expect any unknown Any fields in our protocol buffer. Nevertheless, handle
      // the exception gracefully and, at least, return valid JSON with an id field.
      protoJsonRepresentation =
          "{\"id\" : \"unknown\", \"exception\" : \"InvalidProtocolBufferException\"}\n";
    }
    return protoJsonRepresentation.getBytes(UTF_8);
  }
}
