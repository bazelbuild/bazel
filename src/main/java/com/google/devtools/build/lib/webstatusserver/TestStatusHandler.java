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
import com.google.common.collect.ImmutableList.Builder;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Collection of handlers for displaying the test data.
 */
class TestStatusHandler {
  private StaticResourceHandler frontendHandler;
  private WebStatusBuildLog buildLog;
  private HttpHandler detailsHandler;
  private HttpServer server;
  private ImmutableList<HttpContext> contexts;
  private CommandJsonData commandHandler;
  private Gson gson = new Gson();

  public TestStatusHandler(HttpServer server, WebStatusBuildLog buildLog) {
    Builder<HttpContext> builder = ImmutableList.builder();
    this.buildLog = buildLog;
    this.server = server;
    detailsHandler = new TestStatusResultJsonData(this);
    commandHandler = new CommandJsonData(this);
    frontendHandler = StaticResourceHandler.createFromRelativePath("static/test.html", "text/html");
    builder.add(
        server.createContext("/tests/" + buildLog.getCommandId() + "/details", detailsHandler));
    builder.add(
        server.createContext("/tests/" + buildLog.getCommandId() + "/info", commandHandler));
    builder.add(server.createContext("/tests/" + buildLog.getCommandId(), frontendHandler));
    contexts = builder.build();
  }

  public WebStatusBuildLog getBuildLog() {
    return buildLog;
  }

  
  /**
   *  Serves JSON objects containing command info, which will be rendered by frontend.
   */
  private class CommandJsonData implements HttpHandler {
    private TestStatusHandler testStatusHandler;
    
    public CommandJsonData(TestStatusHandler testStatusHandler) {
      this.testStatusHandler = testStatusHandler;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      exchange.getResponseHeaders().put("Content-Type", ImmutableList.of("application/json"));
      Type commandInfoType = new TypeToken<Map<String, JsonElement>>() {}.getType();
      JsonObject response = gson.toJsonTree(testStatusHandler.buildLog.getCommandInfo(),
          commandInfoType).getAsJsonObject();
      response.addProperty("startTime", testStatusHandler.buildLog.getStartTime());
      response.addProperty("finished", testStatusHandler.buildLog.finished());

      String serializedResponse = response.toString();
      exchange.sendResponseHeaders(200, serializedResponse.length());
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(serializedResponse.getBytes(StandardCharsets.UTF_8));
      }
    }
  }
  
  /**
   * Serves JSON objects containing test cases, which will be rendered by frontend.
   */
  private class TestStatusResultJsonData implements HttpHandler {
    private TestStatusHandler testStatusHandler;

    public TestStatusResultJsonData(TestStatusHandler testStatusHandler) {
      this.testStatusHandler = testStatusHandler;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      Map<String, JsonObject> testInfo = testStatusHandler.buildLog.getTestCases();
      exchange.getResponseHeaders().put("Content-Type", ImmutableList.of("application/json"));
      JsonObject response = new JsonObject();
      for (Entry<String, JsonObject> testCase : testInfo.entrySet()) {
        response.add(testCase.getKey(), testCase.getValue());
      }

      String serializedResponse = response.toString();
      exchange.sendResponseHeaders(200, serializedResponse.length());
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(serializedResponse.getBytes(StandardCharsets.UTF_8));
      }
    }
  }

  /**
   * Adds another URI for existing test data. If specified URI is already used by some other 
   * handler, the previous handler will be removed.
   */
  public void overrideURI(String uri) {
    String detailsPath = uri + "/details";
    String commandPath = uri + "/info";
    try {
      this.server.removeContext(detailsPath);
      this.server.removeContext(commandPath);
    } catch (IllegalArgumentException e) {
      // There was nothing to remove, so proceed with creation (unfortunately the server api doesn't
      // have "hasContext" method)
    }
    this.server.createContext(detailsPath, this.detailsHandler); 
    this.server.createContext(commandPath, this.commandHandler);   
  }
  
  /**
   * Deregisters all the handlers associated with the test.
   */
  public void deregister() {
    for (HttpContext c : this.contexts) {
      this.server.removeContext(c);
    }
  }
}

