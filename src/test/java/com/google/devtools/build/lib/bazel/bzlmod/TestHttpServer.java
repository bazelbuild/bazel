// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.sun.net.httpserver.HttpServer;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;
import org.junit.rules.ExternalResource;

/** A fake HTTP server for testing. */
public class TestHttpServer extends ExternalResource {
  private static final Joiner JOINER = Joiner.on('\n');
  private HttpServer server;
  private String authToken;

  public TestHttpServer(String authToken) {
    this.authToken = authToken;
  }

  public TestHttpServer() {}

  @Override
  protected void before() throws Throwable {
    server = HttpServer.create(new InetSocketAddress(0), 0);
  }

  @Override
  protected void after() {
    server.stop(0);
  }

  public void start() {
    server.start();
  }

  public void serve(String path, byte[] bytes, boolean useAuth) {
    server.createContext(
        path,
        exchange -> {
          if (useAuth) {
            List<String> tokens = exchange.getRequestHeaders().get("Authorization");
            if (tokens == null || tokens.isEmpty() || !authToken.equals(tokens.get(0))) {
              exchange.sendResponseHeaders(401, -1);
              return;
            }
          }
          exchange.sendResponseHeaders(200, bytes.length);
          try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
          }
        });
  }

  public void serve(String path, byte[] bytes) {
    serve(path, bytes, false);
  }

  public void serve(String path, String... lines) {
    serve(path, JOINER.join(lines).getBytes(UTF_8));
  }

  public void unserve(String path) {
    server.removeContext(path);
  }

  public String getUrl() throws MalformedURLException {
    return new URL("http", "[::1]", server.getAddress().getPort(), "").toString();
  }
}
