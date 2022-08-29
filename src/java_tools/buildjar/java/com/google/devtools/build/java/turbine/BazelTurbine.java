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

package com.google.devtools.build.java.turbine;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import com.google.devtools.build.lib.worker.ProtoWorkerMessageProcessor;
import com.google.devtools.build.lib.worker.WorkRequestHandler;
import com.google.devtools.build.lib.worker.WorkRequestHandler.WorkRequestHandlerBuilder;
import com.google.turbine.main.Main;
import com.google.turbine.diag.TurbineError;

/** The Turbine wrapper called by bazel, with support for persistent workers. */
public class BazelTurbine {

  /** The main method of the BazelTurbine. */
  public static void main(String[] args) {
    BazelTurbine builder = new BazelTurbine();
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      WorkRequestHandler workerHandler =
          new WorkRequestHandlerBuilder(
                  builder::build,
                  System.err,
                  new ProtoWorkerMessageProcessor(System.in, System.out))
              .setCpuUsageBeforeGc(Duration.ofSeconds(10))
              .build();
      int exitCode = 1;
      try {
        workerHandler.processRequests();
        exitCode = 0;
      } catch (IOException e) {
        System.err.println(e.getMessage());
      } finally {
        // Prevent hanging threads from keeping the worker alive.
        System.exit(exitCode);
      }
    } else {
      PrintWriter pw =
          new PrintWriter(new OutputStreamWriter(System.err, Charset.defaultCharset()));
      int returnCode;
      try {
        returnCode = builder.build(Arrays.asList(args), pw);
      } finally {
        pw.flush();
      }
      System.exit(returnCode);
    }
  }

  public int build(List<String> args, PrintWriter pw) {
    try {
      Main.compile(args.toArray(new String[0]));
      return 0;
    } catch (TurbineError e) {
      System.err.println(e.getMessage());
      return 1;
    } catch (Throwable turbineCrash) {
      turbineCrash.printStackTrace();
      return 1;
    }
  }
}
