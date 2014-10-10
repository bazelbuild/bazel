// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.webstatusserver;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.blaze.BlazeModule;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.BlazeServerStartupOptions;
import com.google.devtools.build.lib.blaze.BlazeVersionInfo;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import org.joda.time.DateTime;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * Web server for monitoring blaze commands status.
 */
public class WebStatusServerModule extends BlazeModule {
  private static final String LAST_TEST_URI = "/tests/last";
  private HttpServer server;
  private boolean running = false;
  private BlazeServerStartupOptions serverOptions;
  private RawDataHandler lastCommandHandler;
  private static final Logger LOG =
      Logger.getLogger(WebStatusServerModule.class.getCanonicalName());
  private int port;
  // TODO(bazel-team): this list will grow infinitely; at some point old tests should be removed
  // (similarly, old TestStatusHandler should get deregistered from the server at some point)
  private List<WebStatusBuildLog> testsRun = new ArrayList<>();
  private WebStatusBuildLog currentBuild;
  @SuppressWarnings("unused")
  private WebStatusEventCollector collector;
  @SuppressWarnings("unused")
  private IndexPageHandler indexHandler;
  private int commandsRun = 0;
  @Override
  public Iterable<Class<? extends OptionsBase>> getStartupOptions() {
    return ImmutableList.<Class<? extends OptionsBase>>of(BlazeServerStartupOptions.class);
  }

  @Override
  public void blazeStartup(OptionsProvider startupOptions, BlazeVersionInfo versionInfo,
      UUID instanceId, BlazeDirectories directories, Clock clock) throws AbruptExitException {
    serverOptions = startupOptions.getOptions(BlazeServerStartupOptions.class);
    if (serverOptions.useWebStatusServer <= 0) {
      LOG.info("web status server disabled");
      return;
    }
    port = serverOptions.useWebStatusServer;
    try {
      server = HttpServer.create(new InetSocketAddress(port), 0);
      serveStaticContent();
      lastCommandHandler = new RawDataHandler("No commands ran yet.");
      server.createContext("/last", lastCommandHandler);
      server.setExecutor(null);
      server.start();
      indexHandler = new IndexPageHandler(server, this.testsRun);
      running = true;
      LOG.info("Running web status server on port " + port);
    } catch (IOException e) {
      // TODO(bazel-team): Display information about why it failed
      running = false;
      LOG.warning("Unable to run web status server on port " + port);
    }
  }

  @Override
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command) throws AbruptExitException {
    if (!running) {
      return;
    }

    currentBuild = new WebStatusBuildLog();
    collector = new WebStatusEventCollector(blazeRuntime.getEventBus(), currentBuild);
    DateTime currentTime = new DateTime();
    lastCommandHandler.response = "Starting command...\n";
    lastCommandHandler.buildLog = currentBuild;
    lastCommandHandler.command = command;
    lastCommandHandler.startTime = currentTime;

    // TODO(bazel-team): store the tests and cleanup handlers that are not needed anymore (eg. keep
    // only last 1000 tests)
    TestStatusHandler lastTest = new TestStatusHandler(server, commandsRun, currentBuild);
    lastTest.overrideURI(LAST_TEST_URI);
    testsRun.add(currentBuild);
    commandsRun += 1;
  }

  @Override
  public void afterCommand() {
    if (!running) {
      return;
    }
    DateTime currentTime = new DateTime();
    lastCommandHandler.response = "Command finished...\n";
    lastCommandHandler.endTime = currentTime;

    currentBuild = null;
  }

  private void serveStaticContent() {
    StaticResourceHandler testjs =
        StaticResourceHandler.createFromRelativePath("static/test.js", "application/javascript");
    StaticResourceHandler indexjs =
        StaticResourceHandler.createFromRelativePath("static/index.js", "application/javascript");
    StaticResourceHandler d3 = StaticResourceHandler.createFromAbsolutePath(
        "/third_party/javascript/d3/d3-js.js", "application/javascript");
    StaticResourceHandler jquery = StaticResourceHandler.createFromAbsolutePath(
        "/third_party/javascript/jquery/v2_0_3/jquery_uncompressed.jslib",
        "application/javascript");
    StaticResourceHandler testFrontend =
        StaticResourceHandler.createFromRelativePath("static/test.html", "text/html");

    server.createContext("/js/test.js", testjs);
    server.createContext("/js/index.js", indexjs);
    server.createContext("/js/lib/d3.js", d3);
    server.createContext("/js/lib/jquery.js", jquery);
    server.createContext(LAST_TEST_URI, testFrontend);
  }

  /**
   *
   * Dumps data collected by server.
   */
  private static class RawDataHandler implements HttpHandler {
    public DateTime endTime;
    private DateTime startTime;
    private Command command;
    private WebStatusBuildLog buildLog;
    private String response;

    private RawDataHandler(String response) {
      this.response = response;
    }

    @Override
    public void handle(HttpExchange t) throws IOException {
      StringBuilder builder = new StringBuilder(response);
      if (startTime != null) {
        builder.append(startTime.toString());
      }
      if (command != null) {
        builder.append(command.toString());
      }
      if (buildLog != null) {
        builder.append(buildLog.getCommandInfo().toString());
      }
      if (endTime != null) {
        builder.append(endTime.toString());
      }
      String fullResponse = builder.toString();
      t.sendResponseHeaders(200, fullResponse.length());
      OutputStream os = t.getResponseBody();
      os.write(fullResponse.getBytes());
      os.close();
    }
  }
}
