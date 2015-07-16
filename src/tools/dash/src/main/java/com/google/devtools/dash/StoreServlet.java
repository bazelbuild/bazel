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
  private DatastoreService datastore;

  public StoreServlet() {
    super();
    datastore = DatastoreServiceFactory.getDatastoreService();
  }

  @Override
  public void doPost(HttpServletRequest req, HttpServletResponse response) throws IOException {
    DashRequest request;
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
}
