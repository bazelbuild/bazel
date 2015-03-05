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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * Web server for monitoring blaze commands status.
 */
public class WebStatusServerModule extends BlazeModule {
  static final String LAST_TEST_URI = "/tests/last";
  // 100 is an arbitrary limit; it seems like a reasonable size for history and it's okay to change
  // it
  private static final int MAX_TESTS_STORED = 100;

  private HttpServer server;
  private boolean running = false;
  private BlazeServerStartupOptions serverOptions;
  private static final Logger LOG =
      Logger.getLogger(WebStatusServerModule.class.getCanonicalName());
  private int port;
  private LinkedList<TestStatusHandler> testsRan = new LinkedList<>();
  @SuppressWarnings("unused")
  private WebStatusEventCollector collector;
  @SuppressWarnings("unused")
  private IndexPageHandler indexHandler;

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
      TextHandler lastCommandHandler = new TextHandler("No commands ran yet.");
      server.createContext("/last", lastCommandHandler);
      server.setExecutor(null);
      server.start();
      indexHandler = new IndexPageHandler(server, this.testsRan);
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
    collector =
        new WebStatusEventCollector(blazeRuntime.getEventBus(), blazeRuntime.getReporter(), this);
  }

  public void commandStarted() {
    WebStatusBuildLog currentBuild = collector.getBuildLog();

    if (testsRan.size() == MAX_TESTS_STORED) {
      TestStatusHandler oldestTest = testsRan.removeLast();
      oldestTest.deregister();
    }

    TestStatusHandler lastTest = new TestStatusHandler(server, currentBuild);
    testsRan.add(lastTest);

    lastTest.overrideURI(LAST_TEST_URI);
  }

  private void serveStaticContent() {
    StaticResourceHandler testjs =
        StaticResourceHandler.createFromRelativePath("static/test.js", "application/javascript");
    StaticResourceHandler indexjs =
        StaticResourceHandler.createFromRelativePath("static/index.js", "application/javascript");
    StaticResourceHandler style =
        StaticResourceHandler.createFromRelativePath("static/style.css", "text/css");
    StaticResourceHandler d3 = StaticResourceHandler.createFromAbsolutePath(
        "third_party/javascript/d3/d3-js.js", "application/javascript");
    StaticResourceHandler jquery = StaticResourceHandler.createFromAbsolutePath(
        "third_party/javascript/jquery/v2_0_3/jquery_uncompressed.jslib",
        "application/javascript");
    StaticResourceHandler testFrontend =
        StaticResourceHandler.createFromRelativePath("static/test.html", "text/html");

    server.createContext("/css/style.css", style);
    server.createContext("/js/test.js", testjs);
    server.createContext("/js/index.js", indexjs);
    server.createContext("/js/lib/d3.js", d3);
    server.createContext("/js/lib/jquery.js", jquery);
    server.createContext(LAST_TEST_URI, testFrontend);
  }

  private static class TextHandler implements HttpHandler {
    private String response;

    private TextHandler(String response) {
      this.response = response;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      exchange.getResponseHeaders().put("Content-Type", ImmutableList.of("text/plain"));
      exchange.sendResponseHeaders(200, response.length());
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(response.getBytes(StandardCharsets.UTF_8));
      }
    }
  }

  public int getPort() {
    return port;
  }
}

