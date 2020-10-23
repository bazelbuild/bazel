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
import com.google.devtools.build.lib.worker.ExampleWorkerMultiplexerOptions.ExampleWorkMultiplexerOptions;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** An example implementation of a worker process that is used for integration tests. */
public class ExampleWorkerMultiplexer {

  static final Pattern FLAG_FILE_PATTERN = Pattern.compile("(?:@|--?flagfile=)(.+)");

  // Creating Executor Service with a thread pool of Size 3.
  static final int CONCURRENT_THREAD_NUMBER = 3;

  // A UUID that uniquely identifies this running worker process.
  static final UUID WORKER_UUID = UUID.randomUUID();

  // A counter that increases with each work unit processed.
  static int workUnitCounter = 1;

  static int counterOutput = workUnitCounter;

  static Semaphore protectResponse = new Semaphore(1);

  // Keep state across multiple builds.
  static final LinkedHashMap<String, String> inputs = new LinkedHashMap<>();

  private ExampleWorkerMultiplexer() {}

  public static void main(String[] args) throws Exception {
    if (ImmutableSet.copyOf(args).contains("--persistent_worker")) {
      OptionsParser parser =
          OptionsParser.builder()
              .optionsClasses(ExampleWorkerMultiplexerOptions.class)
              .allowResidue(false)
              .build();
      parser.parse(args);
      ExampleWorkerMultiplexerOptions workerOptions =
          parser.getOptions(ExampleWorkerMultiplexerOptions.class);
      Preconditions.checkState(workerOptions.persistentWorker);

      runPersistentWorker(workerOptions);
    } else {
      // This is a single invocation of the example that exits after it processed the request.
      processRequest(parserHelper(ImmutableList.copyOf(args)));
    }
  }

  private static void runPersistentWorker(ExampleWorkerMultiplexerOptions workerOptions)
      throws IOException, ExecutionException, InterruptedException {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    ExecutorService executorService = Executors.newFixedThreadPool(CONCURRENT_THREAD_NUMBER);
    List<Future<?>> results = new ArrayList<>();

    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);
        int requestId = request.getRequestId();
        if (request == null) {
          break;
        }

        inputs.clear();
        for (Input input : request.getInputsList()) {
          inputs.put(input.getPath(), input.getDigest().toStringUtf8());
        }

        // If true, returns corrupt responses instead of correct protobufs.
        boolean poisoned = false;
        if (workerOptions.poisonAfter > 0 && workUnitCounter > workerOptions.poisonAfter) {
          poisoned = true;
        }

        if (poisoned && workerOptions.hardPoison) {
          System.err.println("I'm a very poisoned worker and will just crash.");
          System.exit(1);
        } else {
          int exitCode = 0;
          try {
            OptionsParser parser = parserHelper(request.getArgumentsList());
            ExampleWorkMultiplexerOptions options =
                parser.getOptions(ExampleWorkMultiplexerOptions.class);
            if (options.writeCounter) {
              counterOutput = workUnitCounter++;
            }
            results.add(
                executorService.submit(
                    createTask(originalStdOut, originalStdErr, requestId, parser, poisoned)));
          } catch (Exception e) {
            e.printStackTrace();
            exitCode = 1;
            WorkResponse.newBuilder()
                .setRequestId(requestId)
                .setOutput(new ByteArrayOutputStream().toString())
                .setExitCode(exitCode)
                .build()
                .writeDelimitedTo(System.out);
          }
        }

        if (workerOptions.exitAfter > 0 && workUnitCounter > workerOptions.exitAfter) {
          System.in.close();
        }
      } finally {
        // Be a good worker process and consume less memory when idle.
        System.gc();
      }
    }

    for (Future<?> result : results) {
      result.get();
    }
  }

  private static OptionsParser parserHelper(List<String> args) throws Exception {
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
        OptionsParser.builder()
            .optionsClasses(ExampleWorkMultiplexerOptions.class)
            .allowResidue(true)
            .build();
    parser.parse(expandedArgs.build());

    return parser;
  }

  private static Runnable createTask(
      PrintStream originalStdOut,
      PrintStream originalStdErr,
      int requestId,
      OptionsParser parser,
      boolean poisoned) {
    return () -> {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      int exitCode = 0;

      try {
        try (PrintStream ps = new PrintStream(baos)) {
          System.setOut(ps);
          System.setErr(ps);

          if (poisoned) {
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
              processRequest(parser);
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
          protectResponse.acquire();
          WorkResponse.newBuilder()
              .setRequestId(requestId)
              .setOutput(baos.toString())
              .setExitCode(exitCode)
              .build()
              .writeDelimitedTo(System.out);
          protectResponse.release();
        }
        System.out.flush();
      } catch (IOException | InterruptedException e) {
        throw new IllegalStateException(e);
      }
    };
  }

  private static void processRequest(OptionsParser parser) throws Exception {
    ExampleWorkMultiplexerOptions options = parser.getOptions(ExampleWorkMultiplexerOptions.class);

    List<String> outputs = new ArrayList<>();

    if (options.delay) {
      Integer randomDelay = new Random().nextInt(200) + 100;
      TimeUnit.MILLISECONDS.sleep(randomDelay);
      outputs.add("DELAY " + randomDelay + " MILLISECONDS");
    }

    if (options.writeUUID) {
      outputs.add("UUID " + WORKER_UUID.toString());
    }

    if (options.writeCounter) {
      outputs.add("COUNTER " + counterOutput);
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
