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
import com.google.devtools.build.lib.syntax.Label;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Collection of handlers for displaying the test data.
 */
class TestStatusHandler {
  private TestStatusSummaryJsonData summaryHandler;
  private StaticResourceHandler frontendHandler;
  private WebStatusBuildLog buildLog;
  private HttpHandler detailsHandler;

  public TestStatusHandler(HttpServer server, int commandId, WebStatusBuildLog buildLog) {
    this.buildLog = buildLog;
    summaryHandler = new TestStatusSummaryJsonData(this);
    detailsHandler = new TestStatusResultJsonData(this);
    frontendHandler = StaticResourceHandler.createFromRelativePath("static/test.html", "text/html");
    server.createContext("/tests/" + commandId + "/details", detailsHandler);
    server.createContext("/tests/" + commandId + "/data", summaryHandler);
    server.createContext("/tests/" + commandId, frontendHandler);
  }

  /**
   *
   * Serves JSON objects containing test summaries, which will be rendered by frontend.
   */
  private class TestStatusSummaryJsonData implements HttpHandler {
    private TestStatusHandler testStatusHandler;

    public TestStatusSummaryJsonData(TestStatusHandler testStatusHandler) {
      this.testStatusHandler = testStatusHandler;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      Map<Label, JsonObject> testInfo = testStatusHandler.buildLog.getTestSummaries();
      exchange.getResponseHeaders().put("Content-Type", ImmutableList.of("application/json"));
      JsonArray response = new JsonArray();
      for (Entry<Label, JsonObject> testSummary : testInfo.entrySet()) {
        JsonObject serialized = new JsonObject();

        // Copy over to avoid messing the original data
        // TODO(marcinf): make buildLog.getTests() return a fresh copy of data
        for (Entry<String, JsonElement> entry : testSummary.getValue().entrySet()) {
          serialized.add(entry.getKey(), entry.getValue());
        }

        serialized.addProperty("name", testSummary.getKey().toShorthandString());
        response.add(serialized);
      }
      String serializedResponse = response.toString();
      exchange.sendResponseHeaders(200, serializedResponse.length());
      OutputStream os = exchange.getResponseBody();
      os.write(serializedResponse.getBytes());
      os.close();
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
     OutputStream os = exchange.getResponseBody();
     os.write(serializedResponse.getBytes());
     os.close();
   }
 }
}
