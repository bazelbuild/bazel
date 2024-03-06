// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.MalformedJsonException;
import com.google.protobuf.ByteString;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.Printer;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.IOException;
import java.util.List;

/** Implementation of the Worker Protocol using JSON to communicate with Bazel. */
public final class JsonWorkerMessageProcessor implements WorkRequestHandler.WorkerMessageProcessor {
  /** Reader for reading the WorkResponse. */
  private final JsonReader reader;
  /** Printer for printing the WorkRequest. */
  private final Printer jsonPrinter;
  /** Writer for writing the WorkRequest to the worker. */
  private final BufferedWriter jsonWriter;

  /** Constructs a {@code WorkRequestHandler} that reads and writes JSON. */
  public JsonWorkerMessageProcessor(JsonReader reader, BufferedWriter jsonWriter) {
    this.reader = reader;
    reader.setLenient(true);
    this.jsonWriter = jsonWriter;
    jsonPrinter =
        JsonFormat.printer().omittingInsignificantWhitespace().includingDefaultValueFields();
  }

  private static ImmutableList<String> readArguments(JsonReader reader) throws IOException {
    reader.beginArray();
    ImmutableList.Builder<String> argumentsBuilder = ImmutableList.builder();
    while (reader.hasNext()) {
      argumentsBuilder.add(reader.nextString());
    }
    reader.endArray();
    return argumentsBuilder.build();
  }

  private static ImmutableList<Input> readInputs(JsonReader reader) throws IOException {
    reader.beginArray();
    ImmutableList.Builder<Input> inputsBuilder = ImmutableList.builder();
    while (reader.hasNext()) {
      String digest = null;
      String path = null;

      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        switch (name) {
          case "digest":
            if (digest != null) {
              throw new IOException("Input cannot have more than one digest");
            }
            digest = reader.nextString();
            break;
          case "path":
            if (path != null) {
              throw new IOException("Input cannot have more than one path");
            }
            path = reader.nextString();
            break;
          default:
            // As per https://bazel.build/docs/creating-workers#work-responses,
            // unknown fields are ignored.
            reader.skipValue();
            break;
        }
      }
      reader.endObject();
      Input.Builder inputBuilder = Input.newBuilder();
      if (digest != null) {
        inputBuilder.setDigest(ByteString.copyFromUtf8(digest));
      }
      if (path != null) {
        inputBuilder.setPath(path);
      }
      inputsBuilder.add(inputBuilder.build());
    }
    reader.endArray();
    return inputsBuilder.build();
  }

  @Override
  public WorkRequest readWorkRequest() throws IOException {
    List<String> arguments = null;
    List<Input> inputs = null;
    Integer requestId = null;
    Integer verbosity = null;
    String sandboxDir = null;
    try {
      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        switch (name) {
          case "arguments":
            if (arguments != null) {
              throw new IOException("WorkRequest cannot have more than one 'arguments' field");
            }
            arguments = readArguments(reader);
            break;
          case "inputs":
            if (inputs != null) {
              throw new IOException("WorkRequest cannot have more than one 'inputs' field");
            }
            inputs = readInputs(reader);
            break;
          case "requestId":
            if (requestId != null) {
              throw new IOException("WorkRequest cannot have more than one requestId");
            }
            requestId = reader.nextInt();
            break;
          case "verbosity":
            if (verbosity != null) {
              throw new IOException("Work response cannot have more than one verbosity");
            }
            verbosity = reader.nextInt();
            break;
          case "sandboxDir":
            if (sandboxDir != null) {
              throw new IOException("Work response cannot have more than one sandboxDir");
            }
            sandboxDir = reader.nextString();
            break;
          default:
            // As per https://bazel.build/docs/creating-workers#work-responses,
            // unknown fields are ignored.
            reader.skipValue();
            break;
        }
      }
      reader.endObject();
    } catch (MalformedJsonException | IllegalStateException | EOFException e) {
      throw new IOException(e);
    }

    WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
    if (arguments != null) {
      requestBuilder.addAllArguments(arguments);
    }
    if (inputs != null) {
      requestBuilder.addAllInputs(inputs);
    }
    if (requestId != null) {
      requestBuilder.setRequestId(requestId);
    }
    if (verbosity != null) {
      requestBuilder.setVerbosity(verbosity);
    }
    if (sandboxDir != null) {
      requestBuilder.setSandboxDir(sandboxDir);
    }
    return requestBuilder.build();
  }

  @Override
  public void writeWorkResponse(WorkResponse response) throws IOException {
    jsonPrinter.appendTo(response, jsonWriter);
    jsonWriter.flush();
  }

  @Override
  public void close() throws IOException {
    jsonWriter.close();
  }
}
