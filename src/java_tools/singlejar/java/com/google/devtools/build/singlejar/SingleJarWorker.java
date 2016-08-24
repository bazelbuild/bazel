// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * A blaze worker to run {@link SingleJar} in a warm JVM process.
 */
public class SingleJarWorker {

  public static void main(String[] args) {
    new SingleJarWorker().runWorker();
  }

  private PrintStream originalStdout;
  private PrintStream originalSterr;
  private ByteArrayOutputStream stdoutAndStderr;

  protected void runWorker() {
    trapOutputs();

    try {
      dispatchWorkRequestsForever();
    } catch (IOException e) {
      // IOException will only occur if System.in has been closed
      // In that case we silently exit our process
    }
  }

  private void trapOutputs() {
    originalStdout = System.out;
    originalSterr = System.err;
    stdoutAndStderr = new ByteArrayOutputStream();
    System.setErr(new PrintStream(stdoutAndStderr, true));
    System.setOut(new PrintStream(stdoutAndStderr, true));
  }

  private void dispatchWorkRequestsForever() throws IOException {
    while (true) {
      WorkRequest workRequest = WorkRequest.parseDelimitedFrom(System.in);

      String[] args = workRequest.getArgumentsList().toArray(new String[0]);

      int returnCode = runSingleJar(args);

      outputResult(returnCode);
    }
  }

  private void outputResult(int returnCode) throws IOException {
    WorkResponse.newBuilder()
        .setExitCode(returnCode)
        .setOutput(new String(stdoutAndStderr.toByteArray(), StandardCharsets.UTF_8))
        .build()
        .writeDelimitedTo(originalStdout);

    // Reset output streams, we are not simply calling reset on the BAOS since this will
    // still keep the full buffer allocated.
    stdoutAndStderr = new ByteArrayOutputStream();
    System.setErr(new PrintStream(stdoutAndStderr, true));
    System.setOut(new PrintStream(stdoutAndStderr, true));
  }

  private int runSingleJar(String[] args) {
    try {
      return singleRun(args);
    } catch (IOException e) {
      // Some IO failures are okay no need to quit the worker
      System.err.println("SingleJar threw exception : " + e.getMessage());
      return 1;
    } catch (Exception e) {
      // We had an actual unexpected error, lets quit the worker
      originalSterr.println("SingleJar threw an unexpected exception : " + e.getMessage());
      e.printStackTrace(originalSterr);
      System.exit(1);
      return 1;
    }
  }

  protected int singleRun(String[] args) throws Exception {
    return SingleJar.singleRun(args);
  }
}
