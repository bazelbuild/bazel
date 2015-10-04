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

package com.google.devtools.dash;

import com.google.appengine.api.datastore.DatastoreService;
import com.google.appengine.api.datastore.DatastoreServiceFactory;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Handles storing a test result.
 */
public class StoreServlet extends HttpServlet {
  private static final String DASH_SECRET_HEADER = "bazel-dash-secret";
  private static final String SECRET_PARAMETER = "BAZEL_DASH_SECRET";

  private DatastoreService datastore;

  public StoreServlet() {
    super();
    datastore = DatastoreServiceFactory.getDatastoreService();
  }

  @Override
  public void doPost(HttpServletRequest req, HttpServletResponse response) throws IOException {
    DashRequest request;
    if (!doAuthentication(req, response)) {
      return;
    }
    try {
      request = new DashRequest(req);
    } catch (DashRequest.DashRequestException e) {
      response.setContentType("text/json");
      response.getWriter().println(
          "{ \"error\": \"" + e.getMessage().replaceAll("\"", "") + "\" }");
      return;
    }

    datastore.put(request.getEntity());

    response.setContentType("text/json");
    response.getWriter().println("{ \"ok\": true }");
  }

  private boolean doAuthentication(HttpServletRequest req, HttpServletResponse response)
      throws IOException {
    // Authentication using a common secret
    String secret = System.getenv(SECRET_PARAMETER);
    if (secret != null && !secret.isEmpty()) {
      String providedSecret = req.getHeader(DASH_SECRET_HEADER);
      if (providedSecret == null || !secureCompare(secret, providedSecret)) {
        response.sendError(HttpServletResponse.SC_FORBIDDEN);
        return false;
      }
    }
    return true;
  }

  // Constant time string comparison. Assume that v1 and v2 are not null.
  private boolean secureCompare(String v1, String v2) {
    if (v1.length() != v2.length()) {
      return false;
    }

    int diff = 0;
    for (int i = 0; i < v1.length(); i++) {
      diff |= v1.charAt(i) ^ v2.charAt(i);
    }
    return diff == 0;
  }
}
