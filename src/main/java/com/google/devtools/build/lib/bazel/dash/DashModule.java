// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.dash;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.Subscribe;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.dash.DashProtos.BuildData;
import com.google.devtools.build.lib.bazel.dash.DashProtos.BuildData.CommandLine.Option;
import com.google.devtools.build.lib.bazel.dash.DashProtos.BuildData.EnvironmentVar;
import com.google.devtools.build.lib.bazel.dash.DashProtos.BuildData.Target.TestData;
import com.google.devtools.build.lib.bazel.dash.DashProtos.Log;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.rules.test.TestResult;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandStartEvent;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser.UnparsedOptionValueDescription;
import com.google.devtools.common.options.OptionsProvider;
import com.google.protobuf.ByteString;

import org.apache.http.HttpHeaders;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.DefaultHttpClient;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Dashboard for a build.
 */
public class DashModule extends BlazeModule {
  private static final int ONE_MB = 1024 * 1024;

  private Sendable sender;
  private BlazeRuntime runtime;
  private final ExecutorService executorService;
  private BuildData optionsBuildData;

  public DashModule() {
    // Make sure sender != null before we hop on the event bus.
    sender = new NoOpSender();
    executorService = Executors.newFixedThreadPool(5,
        new ThreadFactory() {
          @Override
          public Thread newThread(Runnable runnable) {
            Thread thread = Executors.defaultThreadFactory().newThread(runnable);
            thread.setDaemon(true);
            return thread;
          }
        });
  }

  @Override
  public void beforeCommand(BlazeRuntime runtime, Command command) {
    this.runtime = runtime;
    runtime.getEventBus().register(this);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return (command.name().equals("build") || command.name().equals("test"))
        ? ImmutableList.<Class<? extends OptionsBase>>of(DashOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    DashOptions options = optionsProvider.getOptions(DashOptions.class);
    sender = (options == null || !options.useDash)
      ? new NoOpSender() : new Sender(options.url, runtime, executorService);
    if (optionsBuildData != null) {
      sender.send("options", optionsBuildData);
    }
    optionsBuildData = null;
  }

  @Subscribe
  public void gotOptions(GotOptionsEvent event) {
    BuildData.Builder builder = BuildData.newBuilder();
    BuildData.CommandLine.Builder cmdLineBuilder = BuildData.CommandLine.newBuilder();
    for (UnparsedOptionValueDescription option :
        event.getStartupOptions().asListOfUnparsedOptions()) {
      cmdLineBuilder.addStartupOptions(getOption(option));
    }

    for (UnparsedOptionValueDescription option : event.getOptions().asListOfUnparsedOptions()) {
      if (option.getName().equals("client_env")) {
        String env[] = option.getUnparsedValue().split("=");
        if (env.length == 1) {
          builder.addClientEnv(
              EnvironmentVar.newBuilder().setName(env[0]).setValue("true").build());
        } else if (env.length == 2) {
          builder.addClientEnv(
              EnvironmentVar.newBuilder().setName(env[0]).setValue(env[1]).build());
        }
      } else {
        cmdLineBuilder.addOptions(getOption(option));
      }
    }

    for (String residue : event.getOptions().getResidue()) {
      cmdLineBuilder.addResidue(residue);
    }
    builder.setCommandLine(cmdLineBuilder.build());

    // This can be called before handleOptions, so the BuildData is stored until we know if it
    // should be sent somewhere.
    optionsBuildData = builder.build();
  }

  @Subscribe
  public void commandStartEvent(CommandStartEvent event) {
    BuildData.Builder builder = BuildData.newBuilder()
        .setBuildId(event.getCommandId().toString())
        .setCommandName(event.getCommandName())
        .setWorkingDir(event.getWorkingDirectory().getPathString());
    sender.send("start", builder.build());
  }

  @Subscribe
  public void parsingComplete(TargetParsingCompleteEvent event) {
    BuildData.Builder builder = BuildData.newBuilder();
    for (Target target : event.getTargets()) {
      builder.addTargetsBuilder()
          .setLabel(target.getLabel().toString())
          .setRuleKind(target.getTargetKind()).build();
    }
    sender.send("targets", builder.build());
  }

  @Subscribe
  public void testFinished(TestResult result) {
    BuildData.Builder builder = BuildData.newBuilder();
    BuildData.Target.Builder targetBuilder = BuildData.Target.newBuilder();
    targetBuilder.setLabel(result.getLabel());
    TestData.Builder testDataBuilder = TestData.newBuilder();
    testDataBuilder.setPassed(result.getData().getTestPassed());
    if (!result.getData().getTestPassed()) {
      testDataBuilder.setLog(getLog(result.getTestLogPath().toString()));
    }
    targetBuilder.setTestData(testDataBuilder);
    builder.addTargets(targetBuilder);
    sender.send("test", builder.build());
  }

  private Log getLog(String logPath) {
    Log.Builder builder = Log.newBuilder().setPath(logPath);
    File log = new File(logPath);
    try {
      long fileSize = Files.size(log.toPath());
      if (fileSize > ONE_MB) {
        fileSize = ONE_MB;
        builder.setTruncated(true);
      }
      byte buffer[] = new byte[(int) fileSize];
      try (FileInputStream in = new FileInputStream(log)) {
        ByteStreams.readFully(in, buffer);
      }
      builder.setContents(ByteString.copyFrom(buffer));
    } catch (IOException e) {
      runtime
          .getReporter()
          .getOutErr()
          .printOutLn("Error reading log file " + logPath + ": " + e.getMessage());
      // TODO(kchodorow): add this info to the proto and send.
    }
    return builder.build();
  }

  @Override
  public void blazeShutdown() {
    executorService.shutdownNow();
  }

  private BuildData.CommandLine.Option getOption(UnparsedOptionValueDescription option) {
    Option.Builder optionBuilder = Option.newBuilder();
    optionBuilder.setName(option.getName());
    if (option.getSource() != null) {
      optionBuilder.setSource(option.getSource());
    }
    Object value = option.getUnparsedValue();
    if (value != null) {
      if (value instanceof Iterable<?>) {
        for (Object v : ((Iterable<?>) value)) {
          if (v != null) {
            optionBuilder.addValue(v.toString());
          }
        }
      } else {
        optionBuilder.addValue(value.toString());
      }
    }
    return optionBuilder.build();
  }

  private interface Sendable {
    void send(final String suffix, final BuildData message);
  }

  private static class Sender implements Sendable {
    private final String url;
    private final String buildId;
    private final OutErr outErr;
    private final ExecutorService executorService;

    public Sender(String url, BlazeRuntime runtime, ExecutorService executorService) {
      this.url = url;
      this.buildId = runtime.getCommandId().toString();
      this.outErr = runtime.getReporter().getOutErr();
      this.executorService = executorService;
      runtime
          .getReporter()
          .handle(Event.info("Results are being streamed to " + url + "/result/" + buildId));
    }

    @Override
    public void send(final String suffix, final BuildData message) {
      executorService.submit(new Runnable() {
        @Override
        public void run() {
          HttpClient httpClient = new DefaultHttpClient();
          HttpPost httppost = new HttpPost(url + "/" + suffix + "/" + buildId);
          httppost.setHeader(HttpHeaders.CONTENT_TYPE, "application/x-protobuf");
          httppost.setEntity(new ByteArrayEntity(message.toByteArray()));

          try {
            httpClient.execute(httppost);
          } catch (IOException | IllegalStateException e) {
            // IllegalStateException is thrown if the URL was invalid (e.g., someone passed
            // --dash_url=localhost:8080 instead of --dash_url=http://localhost:8080).
            outErr.printErrLn("Error sending results to " + url + ": " + e.getMessage());
          } catch (Exception e) {
            outErr.printErrLn("Unknown error sending results to " + url + ": " + e.getMessage());
          }
        }
      });
    }
  }

  private static class NoOpSender implements Sendable {
    public NoOpSender() {
    }

    @Override
    public void send(String suffix, BuildData message) {
    }
  }

}
