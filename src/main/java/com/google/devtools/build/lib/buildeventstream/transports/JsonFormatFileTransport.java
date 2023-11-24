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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.gson.GsonBuilder;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.Printer;
import com.google.protobuf.util.JsonFormat.TypeRegistry;
import java.io.BufferedOutputStream;

/**
 * A simple {@link BuildEventTransport} that writes the JSON representation of the protocol-buffer
 * representation of the events to a file.
 */
public final class JsonFormatFileTransport extends FileTransport {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final String UNKNOWN_ANY_TYPE_ERROR_EVENT =
      new GsonBuilder().create().toJson(new UnknownAnyProtoError());

  private final Printer jsonPrinter;

  public JsonFormatFileTransport(
      BufferedOutputStream outputStream,
      BuildEventProtocolOptions options,
      BuildEventArtifactUploader uploader,
      ArtifactGroupNamer namer,
      TypeRegistry typeRegistry,
      BesUploadMode besUploadMode) {
    super(outputStream, options, uploader, namer, besUploadMode);
    jsonPrinter =
        JsonFormat.printer().usingTypeRegistry(typeRegistry).omittingInsignificantWhitespace();
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }

  @Override
  protected byte[] serializeEvent(BuildEventStreamProtos.BuildEvent buildEvent) {
    String protoJsonRepresentation;
    try {
      protoJsonRepresentation = jsonPrinter.print(buildEvent);
    } catch (InvalidProtocolBufferException e) {
      // We don't expect any unknown Any fields in our protocol buffer. Nevertheless, handle
      // the exception gracefully and, at least, return valid JSON with an id field.
      logger.atWarning().withCause(e).log(
          "Failed to serialize to JSON due to Any type resolution failure: %s", buildEvent);
      protoJsonRepresentation = UNKNOWN_ANY_TYPE_ERROR_EVENT;
    }
    return (protoJsonRepresentation + "\n").getBytes(UTF_8);
  }

  /** Error produced when serializing an {@code Any} protobuf whose contained type is unknown. */
  @VisibleForTesting
  static class UnknownAnyProtoError {
    @SuppressWarnings({"FieldCanBeStatic", "unused"}) // Used by Gson formatting; cannot be static
    private final String id = "unknown";

    @SuppressWarnings({"FieldCanBeStatic", "unused"}) // Used by Gson formatting; cannot be static
    private final String exception = "InvalidProtocolBufferException";
  }
}
