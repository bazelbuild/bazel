// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.worker.WorkerProtocol.Input;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.OptionsParser;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * An example implementation of a worker process that is used for integration tests.
 */
public class ExampleWorker {

  static final Pattern FLAG_FILE_PATTERN = Pattern.compile("(?:@|--?flagfile=)(.+)");

  // A UUID that uniquely identifies this running worker process.
  static final UUID workerUuid = UUID.randomUUID();

  // A counter that increases with each work unit processed.
  static int workUnitCounter = 1;

  // If true, returns corrupt responses instead of correct protobufs.
  static boolean poisoned = false;

  // Keep state across multiple builds.
  static final LinkedHashMap<String, String> inputs = new LinkedHashMap<>();

  public static void main(String[] args) throws Exception {
    if (ImmutableSet.copyOf(args).contains("--persistent_worker")) {
      OptionsParser parser =
          OptionsParser.builder()
              .optionsClasses(ExampleWorkerOptions.class)
              .allowResidue(false)
              .build();
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

        inputs.clear();
        for (Input input : request.getInputsList()) {
          inputs.put(input.getPath(), input.getDigest().toStringUtf8());
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        int exitCode = 0;

        try (PrintStream ps = new PrintStream(baos)) {
          System.setOut(ps);
          System.setErr(ps);

          if (poisoned) {
            if (workerOptions.hardPoison) {
              throw new IllegalStateException("I'm a very poisoned worker and will just crash.");
            }
            System.out.println("I'm a poisoned worker and this is not a protobuf.");
            System.out.println("Here's a fake stack trace for you:");
            System.out.println("    at com.example.Something(Something.java:83)");
            System.out.println("    at java.lang.Thread.run(Thread.java:745)");
            System.out.print("And now, 8k of random bytes: ");
            byte[] b = new byte[8192];
            new Random().nextBytes(b);
            System.out.write(b);
          } else {
            try {
              processRequest(request.getArgumentsList());
            } catch (Exception e) {
              e.printStackTrace();
              exitCode = 1;
            }
          }
        } finally {
          System.setOut(originalStdOut);
          System.setErr(originalStdErr);
        }

        if (poisoned) {
          baos.writeTo(System.out);
        } else {
          WorkResponse.newBuilder()
              .setOutput(baos.toString())
              .setExitCode(exitCode)
              .build()
              .writeDelimitedTo(System.out);
        }
        System.out.flush();

        if (workerOptions.exitAfter > 0 && workUnitCounter > workerOptions.exitAfter) {
          return;
        }

        if (workerOptions.poisonAfter > 0 && workUnitCounter > workerOptions.poisonAfter) {
          poisoned = true;
        }
      } finally {
        // Be a good worker process and consume less memory when idle.
        System.gc();
      }
    }
  }

  private static void processRequest(List<String> args) throws Exception {
    ImmutableList.Builder<String> expandedArgs = ImmutableList.builder();
    for (String arg : args) {
      Matcher flagFileMatcher = FLAG_FILE_PATTERN.matcher(arg);
      if (flagFileMatcher.matches()) {
        expandedArgs.addAll(Files.readAllLines(Paths.get(flagFileMatcher.group(1)), UTF_8));
      } else {
        expandedArgs.add(arg);
      }
    }

    OptionsParser parser =
        OptionsParser.builder().optionsClasses(ExampleWorkOptions.class).allowResidue(true).build();
    parser.parse(expandedArgs.build());
    ExampleWorkOptions options = parser.getOptions(ExampleWorkOptions.class);

    List<String> outputs = new ArrayList<>();

    if (options.writeUUID) {
      outputs.add("UUID " + workerUuid.toString());
    }

    if (options.writeCounter) {
      outputs.add("COUNTER " + workUnitCounter++);
    }

    String residueStr = Joiner.on(' ').join(parser.getResidue());
    if (options.uppercase) {
      residueStr = residueStr.toUpperCase();
    }
    outputs.add(residueStr);

    if (options.printInputs) {
      for (Map.Entry<String, String> input : inputs.entrySet()) {
        outputs.add("INPUT " + input.getKey() + " " + input.getValue());
      }
    }

    if (options.printEnv) {
      for (Map.Entry<String, String> entry : System.getenv().entrySet()) {
        outputs.add(entry.getKey() + "=" + entry.getValue());
      }
    }

    String outputStr = Joiner.on('\n').join(outputs);
    if (options.outputFile.isEmpty()) {
      System.out.println(outputStr);
    } else {
      try (PrintStream outputFile = new PrintStream(options.outputFile)) {
        outputFile.println(outputStr);
      }
    }
  }
}
