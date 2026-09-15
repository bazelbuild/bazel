// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Implementation of the Worker Protocol using Proto to communicate with Bazel. */
public final class ProtoWorkerMessageProcessor
    implements WorkRequestHandler.WorkerMessageProcessor {

  /** This worker's stdin. */
  private final InputStream stdin;

  /** This worker's stdout. Only {@link WorkRequest}s should be written here. */
  private final OutputStream stdout;

  /** Constructs a {@link WorkRequestHandler} that reads and writes Protocol Buffers. */
  public ProtoWorkerMessageProcessor(InputStream stdin, OutputStream stdout) {
    this.stdin = stdin;
    this.stdout = stdout;
  }

  @Override
  public WorkRequest readWorkRequest() throws IOException {
    return WorkRequest.parseDelimitedFrom(stdin);
  }

  @Override
  public void writeWorkResponse(WorkResponse workResponse) throws IOException {
    try {
      workResponse.writeDelimitedTo(stdout);
    } finally {
      stdout.flush();
    }
  }

  @Override
  public void close() {}
}
