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
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.MalformedJsonException;
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

/** An implementation of a Bazel worker using JSON to communicate with the worker process. */
final class JsonWorkerProtocol implements WorkerProtocolImpl {
  /** Reader for reading the WorkResponse. */
  private final JsonReader reader;
  /** Printer for printing the WorkRequest */
  private final Printer jsonPrinter;
  /** Writer for writing the WorkRequest to the worker */
  private final BufferedWriter jsonWriter;

  JsonWorkerProtocol(OutputStream workersStdin, InputStream workersStdout) {
    jsonPrinter = JsonFormat.printer().omittingInsignificantWhitespace();
    jsonWriter = new BufferedWriter(new OutputStreamWriter(workersStdin, UTF_8));
    reader = new JsonReader(new BufferedReader(new InputStreamReader(workersStdout, UTF_8)));
    reader.setLenient(true);
  }

  @Override
  public void putRequest(WorkRequest request) throws IOException {
    // WorkRequests are serialized according to ndjson spec.
    // https://github.com/ndjson/ndjson-spec
    jsonPrinter.appendTo(request, jsonWriter);
    jsonWriter.append("\n");
    jsonWriter.flush();
  }

  @Override
  public WorkResponse getResponse() throws IOException {
    boolean interrupted = Thread.interrupted();
    try {
      return parseResponse();
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private WorkResponse parseResponse() throws IOException {
    Integer exitCode = null;
    String output = null;
    Integer requestId = null;
    try {
      // After reading at least one ndjson message, gson returns END_DOCUMENT at EOF instead of
      // throwing EOFException. Check for it before beginObject() to signal clean shutdown.
      if (reader.peek() == JsonToken.END_DOCUMENT) {
        return null;
      }
      reader.beginObject();
    } catch (EOFException e) {
      // Clean EOF on an empty stream (no prior messages): the worker closed its stdout before
      // starting any message. Return null to signal shutdown, matching proto behavior.
      return null;
    } catch (MalformedJsonException | IllegalStateException e) {
      throw new IOException("Could not parse json work response", e);
    }
    try {
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
            // As per https://bazel.build/docs/creating-workers#work-responses,
            // unknown fields are ignored.
            reader.skipValue();
        }
      }
      reader.endObject();
    } catch (MalformedJsonException | EOFException | IllegalStateException e) {
      // EOF after beginObject() means a truncated message, not a clean shutdown.
      throw new IOException("Could not parse json work response", e);
    }

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
