package com.google.devtools.build.java.turbine;

import com.google.turbine.main.Main;
import com.google.devtools.build.lib.worker.ProtoWorkerMessageProcessor;
import com.google.devtools.build.lib.worker.WorkRequestHandler;

import java.io.IOException;
import java.io.PrintWriter;
import java.time.Duration;
import java.util.List;

/**
 * A Wrapper for Turbine to support multiplex worker
 */
public class TurbineWorkerWrapper {

  public static void main(String[] args) throws IOException {
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      WorkRequestHandler workerHandler =
          new WorkRequestHandler.WorkRequestHandlerBuilder(
              TurbineWorkerWrapper::turbine,
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
      Main.main(args);
    }
  }

  private static int turbine(List<String> args, PrintWriter pw) {
      try {
          Main.compile(args.toArray(new String[0]));
      } catch (Throwable e) {
          System.err.println(e.getMessage());
          return 1;
      }
      return 0;
  }

}