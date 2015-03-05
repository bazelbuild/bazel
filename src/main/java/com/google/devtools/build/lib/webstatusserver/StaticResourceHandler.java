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
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.util.ResourceFileLoader;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Handler for static resources (JS, html, css...)
 */
public class StaticResourceHandler implements HttpHandler {
  private String response;
  private List<String> contentType;
  private int httpCode;

  public static StaticResourceHandler createFromAbsolutePath(String path, String contentType) {
    return new StaticResourceHandler(path, contentType, true);
  }

  public static StaticResourceHandler createFromRelativePath(String path, String contentType) {
    return new StaticResourceHandler(path, contentType, false);
  }

  private StaticResourceHandler(String path, String contentType, boolean absolutePath) {
    try {
      if (absolutePath) {
        InputStream resourceStream = loadFromAbsolutePath(WebStatusServerModule.class, path);
        response = CharStreams.toString(new InputStreamReader(resourceStream));

      } else {
        response = ResourceFileLoader.loadResource(WebStatusServerModule.class, path);
      }
      httpCode = 200;
    } catch (IOException e) {
      throw new IllegalArgumentException("resource " + path + " not found");
    }
    this.contentType = ImmutableList.of(contentType);
  }

  @Override
  public void handle(HttpExchange exchange) throws IOException {
    exchange.getResponseHeaders().put("Content-Type", contentType);
    exchange.sendResponseHeaders(httpCode, response.length());
    try (OutputStream os = exchange.getResponseBody()) {
      os.write(response.getBytes(StandardCharsets.UTF_8));
    }
  }

  public static InputStream loadFromAbsolutePath(Class<?> loadingClass, String path)
      throws IOException {
    URL resourceUrl = loadingClass.getClassLoader().getResource(path);
    if (resourceUrl == null) {
      throw new IllegalArgumentException("resource " + path + " not found");
    }
    return resourceUrl.openStream();
  }
}
