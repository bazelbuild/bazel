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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.MalformedJsonException;
import com.google.protobuf.ByteString;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.Printer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

/** Sample implementation of the Worker Protocol using JSON to communicate with Bazel. */
public class JsonExampleWorkerProtocolImpl implements ExampleWorkerProtocol {
  private final Printer jsonPrinter =
      JsonFormat.printer().omittingInsignificantWhitespace().includingDefaultValueFields();
  private final JsonReader reader;
  private final BufferedWriter jsonWriter;

  public JsonExampleWorkerProtocolImpl(InputStream stdin, OutputStream stdout) {
    reader = new JsonReader(new BufferedReader(new InputStreamReader(stdin, UTF_8)));
    reader.setLenient(true);
    jsonWriter = new BufferedWriter(new OutputStreamWriter(stdout, UTF_8));
  }

  private static ArrayList<String> readArguments(JsonReader reader) throws IOException {
    reader.beginArray();
    ArrayList<String> arguments = new ArrayList<>();
    while (reader.hasNext()) {
      arguments.add(reader.nextString());
    }
    reader.endArray();
    return arguments;
  }

  private static ArrayList<Input> readInputs(JsonReader reader) throws IOException {
    reader.beginArray();
    ArrayList<Input> inputs = new ArrayList<>();
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
            throw new IOException(name + " is an incorrect field in input");
        }
      }
      reader.endObject();
      inputs.add(
          Input.newBuilder().setDigest(ByteString.copyFromUtf8(digest)).setPath(path).build());
    }
    reader.endArray();
    return inputs;
  }

  @Override
  public WorkRequest readRequest() throws IOException {
    List<String> arguments = null;
    List<Input> inputs = null;
    Integer requestId = null;
    try {
      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        switch (name) {
          case "arguments":
            if (arguments != null) {
              throw new IOException("Work request cannot have more than one list of arguments");
            }
            arguments = readArguments(reader);
            break;
          case "inputs":
            if (inputs != null) {
              throw new IOException("Work request cannot have more than one list of inputs");
            }
            inputs = readInputs(reader);
            break;
          case "requestId":
            if (requestId != null) {
              throw new IOException("Work request cannot have more than one requestId");
            }
            requestId = reader.nextInt();
            break;
          default:
            throw new IOException(name + " is an incorrect field in work request");
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
    return requestBuilder.build();
  }

  @Override
  public void writeResponse(WorkResponse response) throws IOException {
    jsonPrinter.appendTo(response, jsonWriter);
    jsonWriter.flush();
  }

  @Override
  public void close() {
    try {
      jsonWriter.close();
    } catch (IOException e) {
      System.err.printf("Could not close json writer. %s", e);
    }
  }
}
