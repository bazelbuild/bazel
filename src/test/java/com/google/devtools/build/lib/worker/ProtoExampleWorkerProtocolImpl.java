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

import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Sample implementation of the Worker Protocol using Proto to communiocate with Bazel. */
public class ProtoExampleWorkerProtocolImpl implements ExampleWorkerProtocol {
  private final InputStream stdin;
  private final OutputStream stdout;

  public ProtoExampleWorkerProtocolImpl(InputStream stdin, OutputStream stdout) {
    this.stdin = stdin;
    this.stdout = stdout;
  }

  @Override
  public WorkRequest readRequest() throws IOException {
    try {
      return WorkRequest.parseDelimitedFrom(stdin);
    } catch (InvalidProtocolBufferException e) {
      throw new IOException(e);
    }
  }

  @Override
  public void writeResponse(WorkResponse response) throws IOException {
    response.writeDelimitedTo(stdout);
    stdout.flush();
  }

  @Override
  public void close() {}
}
