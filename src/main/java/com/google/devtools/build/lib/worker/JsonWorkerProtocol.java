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

import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.gson.stream.JsonReader;
import com.google.protobuf.util.JsonFormat;
import com.google.protobuf.util.JsonFormat.Printer;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

/** An implementation of a Bazel worker using JSON to communicate with the worker process. */
final class JsonWorkerProtocol implements WorkerProtocolImpl {
  /** Reader for reading the WorkResponse. */
  private final JsonReader reader;
  /** Printer for printing the WorkRequest */
  private final Printer jsonPrinter;
  /** Writer for writing the WorkRequest to the worker */
  private final BufferedWriter jsonWriter;

  JsonWorkerProtocol(InputStream stdin, OutputStream stdout) {
    jsonPrinter = JsonFormat.printer().omittingInsignificantWhitespace();
    jsonWriter = new BufferedWriter(new OutputStreamWriter(stdout, UTF_8));
    reader = new JsonReader(new BufferedReader(new InputStreamReader(stdin, UTF_8)));
    reader.setLenient(true);
  }

  @Override
  public void putRequest(WorkRequest request) throws IOException {
    jsonPrinter.appendTo(request, jsonWriter);
    jsonWriter.flush();
  }

  @Override
  public WorkResponse getResponse() throws IOException {
    Integer exitCode = null;
    String output = null;
    Integer requestId = null;

    reader.beginObject();
    while (reader.hasNext()) {
      String name = reader.nextName();
      switch (name) {
        case "exitCode":
          if (exitCode != null) {
            throw new IOException("Work response cannot have more than one exit code");
          }
          exitCode = reader.nextInt();
          break;
        case "output":
          if (output != null) {
            throw new IOException("Work response cannot have more than one output");
          }
          output = reader.nextString();
          break;
        case "requestId":
          if (requestId != null) {
            throw new IOException("Work response cannot have more than one requestId");
          }
          requestId = reader.nextInt();
          break;
        default:
          throw new IOException(name + " is an incorrect field in work response");
      }
    }
    reader.endObject();

    WorkResponse.Builder responseBuilder = WorkResponse.newBuilder();

    if (exitCode != null) {
      responseBuilder.setExitCode(exitCode);
    }
    if (output != null) {
      responseBuilder.setOutput(output);
    }
    if (requestId != null) {
      responseBuilder.setRequestId(requestId);
    }

    return responseBuilder.build();
  }

  @Override
  public void close() throws IOException {
    reader.close();
    jsonWriter.close();
  }
}
