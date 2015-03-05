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
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Handlers for displaying the index page of server.
 *
 */
public class IndexPageHandler {
  private List<TestStatusHandler> testHandlers = new ArrayList<>();
  private IndexPageJsonData dataHandler;
  private StaticResourceHandler frontendHandler;

  public IndexPageHandler(HttpServer server, List<TestStatusHandler> testHandlers) {
    this.testHandlers = testHandlers;
    this.dataHandler = new IndexPageJsonData(this);
    this.frontendHandler =
        StaticResourceHandler.createFromRelativePath("static/index.html", "text/html");
    server.createContext("/", frontendHandler);
    server.createContext("/tests/list", dataHandler);
  }

  /**
   * Puts data from the build log into json suitable for frontend.
   * 
   */
  private class IndexPageJsonData implements HttpHandler {
    private IndexPageHandler pageHandler;
    private Gson gson = new Gson();
    public IndexPageJsonData(IndexPageHandler indexPageHandler) {
      this.pageHandler = indexPageHandler;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      exchange.getResponseHeaders().put("Content-Type", ImmutableList.of("application/json"));
      JsonArray response = new JsonArray();
      for (TestStatusHandler handler : this.pageHandler.testHandlers) {  
        WebStatusBuildLog buildLog = handler.getBuildLog();
        JsonObject test = new JsonObject();
        test.add("targets",  gson.toJsonTree(buildLog.getTargetList()));
        test.addProperty("startTime", buildLog.getStartTime());
        test.addProperty("finished", buildLog.finished());
        test.addProperty("uuid", buildLog.getCommandId().toString());
        response.add(test);
      }
      String serializedResponse = response.toString();
      exchange.sendResponseHeaders(200, serializedResponse.length());
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(serializedResponse.getBytes(StandardCharsets.UTF_8));
      }
    }
  }
}

