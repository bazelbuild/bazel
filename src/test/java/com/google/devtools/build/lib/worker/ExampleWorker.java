// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.worker.ExampleWorkerOptions.ExampleWorkOptions;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.OptionsParser;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * An example implementation of a worker process that is used for integration tests.
 */
public class ExampleWorker {

  public static void main(String[] args) throws Exception {
    if (ImmutableSet.copyOf(args).contains("--persistent_worker")) {
      OptionsParser parser = OptionsParser.newOptionsParser(ExampleWorkerOptions.class);
      parser.setAllowResidue(false);
      parser.parse(args);
      ExampleWorkerOptions workerOptions = parser.getOptions(ExampleWorkerOptions.class);
      Preconditions.checkState(workerOptions.persistentWorker);

      runPersistentWorker(workerOptions);
    } else {
      // This is a single invocation of the example that exits after it processed the request.
      processRequest(ImmutableList.copyOf(args));
    }
  }

  private static void runPersistentWorker(ExampleWorkerOptions workerOptions) throws IOException {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);
        if (request == null) {
          break;
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        int exitCode = 0;

        try (PrintStream ps = new PrintStream(baos)) {
          System.setOut(ps);
          System.setErr(ps);

          try {
            processRequest(request.getArgumentsList());
          } catch (Exception e) {
            e.printStackTrace();
            exitCode = 1;
          }
        } finally {
          System.setOut(originalStdOut);
          System.setErr(originalStdErr);
        }

        WorkResponse.newBuilder()
            .setOutput(baos.toString())
            .setExitCode(exitCode)
            .build()
            .writeDelimitedTo(System.out);
        System.out.flush();
      } finally {
        // Be a good worker process and consume less memory when idle.
        System.gc();
      }
    }
  }

  private static void processRequest(List<String> args) throws Exception {
    if (args.size() == 1 && args.get(0).startsWith("@")) {
      args = Files.readAllLines(Paths.get(args.get(0).substring(1)), UTF_8);
    }

    OptionsParser parser = OptionsParser.newOptionsParser(ExampleWorkOptions.class);
    parser.setAllowResidue(true);
    parser.parse(args);
    ExampleWorkOptions workOptions = parser.getOptions(ExampleWorkOptions.class);

    List<String> residue = parser.getResidue();
    List<String> outputs = new ArrayList(residue.size());
    for (String arg : residue) {
      String output = arg;
      if (workOptions.uppercase) {
        output = output.toUpperCase();
      }
      outputs.add(output);
    }

    String outputStr = Joiner.on(' ').join(outputs);
    if (workOptions.outputFile.isEmpty()) {
      System.err.println("ExampleWorker: Writing to stdout!");
      System.out.println(outputStr);
    } else {
      System.err.println("ExampleWorker: Writing to file " + workOptions.outputFile);
      try (PrintStream outputFile = new PrintStream(workOptions.outputFile)) {
        outputFile.println(outputStr);
      }
    }
  }
}
