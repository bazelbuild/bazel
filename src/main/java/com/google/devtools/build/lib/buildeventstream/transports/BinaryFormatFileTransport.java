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

import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.protobuf.CodedOutputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.function.Consumer;

/**
 * A simple {@link BuildEventTransport} that writes a varint delimited binary representation of
 * {@link BuildEvent} protocol buffers to a file.
 */
public final class BinaryFormatFileTransport extends FileTransport {
  public BinaryFormatFileTransport(
      BufferedOutputStream outputStream,
      BuildEventProtocolOptions options,
      BuildEventArtifactUploader uploader,
      Consumer<AbruptExitException> abruptExitCallback,
      ArtifactGroupNamer namer) {
    super(outputStream, options, uploader, abruptExitCallback, namer);
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }

  @Override
  protected byte[] serializeEvent(BuildEventStreamProtos.BuildEvent buildEvent) {
    final int size = buildEvent.getSerializedSize();
    ByteArrayOutputStream bos =
        new ByteArrayOutputStream(CodedOutputStream.computeUInt32SizeNoTag(size) + size);
    try {
      buildEvent.writeDelimitedTo(bos);
    } catch (IOException e) {
      throw new RuntimeException(
          "Unexpected error serializing protobuf to in memory outputstream.", e);
    }
    return bos.toByteArray();
  }
}
