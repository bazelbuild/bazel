// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.rules.test.TestResult;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommandStartEvent;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser.UnparsedOptionValueDescription;
import com.google.devtools.common.options.OptionsProvider;
import com.google.protobuf.ByteString;

import org.apache.http.HttpEntity;
import org.apache.http.HttpHeaders;
import org.apache.http.HttpStatus;
import org.apache.http.StatusLine;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Dashboard for a build.
 */
public class DashModule extends BlazeModule {
  private static final int ONE_MB = 1024 * 1024;

  private static final String DASH_SECRET_HEADER = "bazel-dash-secret";

  private Sendable sender;
  private CommandEnvironment env;
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
  public void beforeCommand(Command command, CommandEnvironment env) {
    this.env = env;
    env.getEventBus().register(this);
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
    try {
      sender = (options == null || !options.useDash)
          ? new NoOpSender()
          : new Sender(options.url, options.secret, env, executorService);
    } catch (SenderException e) {
      env.getReporter().handle(e.toEvent());
      sender = new NoOpSender();
    }
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
      env
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

  private static class SenderException extends Exception {
    SenderException(String message, Throwable ex) {
      super(message, ex);
    }

    SenderException(String message) {
      super(message);
    }
    
    Event toEvent() {
      if (getCause() != null) {
        return Event.error(getMessage() + ": " + getCause().getMessage());
      } else {
        return Event.error(getMessage());
      }
    }
  }

  private static class Sender implements Sendable {
    private final URL url;
    private final String buildId;
    private final String secret;
    private final Reporter reporter;
    private final ExecutorService executorService;

    public Sender(String url, String secret,
        CommandEnvironment env, ExecutorService executorService) throws SenderException {
      this.reporter = env.getReporter();
      this.secret = readSecret(secret, reporter);
      try {
        this.url = new URL(url);
        if (!this.secret.isEmpty()) {
          if (!(this.url.getProtocol().equals("https") || this.url.getHost().equals("localhost")
                  || this.url.getHost().matches("^127.0.0.[0-9]+$"))) {
            reporter.handle(Event.warn("Using authentication over unsecure channel, "
                + "consider using HTTPS."));
          }
        }
      } catch (MalformedURLException e) {
        throw new SenderException("Invalid server url " + url, e);
      }
      this.buildId = env.getCommandId().toString();
      this.executorService = executorService;
      sendMessage("test", null); // test connecting to the server.
      reporter.handle(Event.info("Results are being streamed to " + url + "/result/" + buildId));
    }

    private static String readSecret(String secretFile, Reporter reporter) throws SenderException {
      if (secretFile.isEmpty()) {
        return "";
      }
      Path secretPath = new File(secretFile).toPath();
      if (!Files.isReadable(secretPath)) {
        throw new SenderException("Secret file " + secretFile + " doesn't exists or is unreadable");
      }
      try {
        if (Files.getPosixFilePermissions(secretPath).contains(PosixFilePermission.OTHERS_READ)
            || Files.getPosixFilePermissions(secretPath).contains(PosixFilePermission.GROUP_READ)) {
          reporter.handle(Event.warn("Secret file " + secretFile + " is readable by non-owner. "
              + "It is recommended to set its permission to 0600 (read-write only by the owner)."));
        }
        return new String(Files.readAllBytes(secretPath), StandardCharsets.UTF_8).trim();
      } catch (IOException e) {
        throw new SenderException("Invalid secret file " + secretFile, e);
      }
    }

    private void sendMessage(final String suffix, final HttpEntity message)
        throws SenderException {
      HttpParams httpParams = new BasicHttpParams();
      HttpConnectionParams.setConnectionTimeout(httpParams, 5000);
      HttpConnectionParams.setSoTimeout(httpParams, 5000);
      HttpClient httpClient = new DefaultHttpClient(httpParams);

      HttpPost httppost = new HttpPost(url + "/" + suffix + "/" + buildId);
      if (message != null) {
        httppost.setHeader(HttpHeaders.CONTENT_TYPE, "application/x-protobuf");
        httppost.setEntity(message);
      }
      if (!secret.isEmpty()) {
        httppost.setHeader(DASH_SECRET_HEADER, secret);
      }
      StatusLine status;
      try {
        status = httpClient.execute(httppost).getStatusLine();
      } catch (IOException e) {
        throw new SenderException("Error sending results to " + url, e);
      }
      if (status.getStatusCode() == HttpStatus.SC_FORBIDDEN) {
        throw new SenderException("Permission denied while sending results to " + url
            + ". Did you specified --dash_secret?");
      }
    }

    @Override
    public void send(final String suffix, final BuildData message) {
      executorService.submit(new Runnable() {
        @Override
        public void run() {
          try {
            sendMessage(suffix, new ByteArrayEntity(message.toByteArray()));
          } catch (SenderException ex) {
            reporter.handle(ex.toEvent());
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
