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

import com.google.appengine.api.datastore.Blob;
import com.google.appengine.api.datastore.DatastoreService;
import com.google.appengine.api.datastore.DatastoreServiceFactory;
import com.google.appengine.api.datastore.Entity;
import com.google.appengine.api.datastore.PreparedQuery;
import com.google.appengine.api.datastore.Query;
import com.google.appengine.api.datastore.Query.FilterOperator;
import com.google.appengine.api.datastore.Query.FilterPredicate;
import com.google.common.html.HtmlEscapers;
import com.google.devtools.build.lib.bazel.dash.DashProtos.BuildData;

import org.apache.velocity.Template;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.VelocityEngine;

import java.io.IOException;
import java.io.StringWriter;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Handles HTTP gets of builds/tests.
 */
public class BuildViewServlet extends HttpServlet {
  private DatastoreService datastore;

  public BuildViewServlet() {
    super();
    datastore = DatastoreServiceFactory.getDatastoreService();
  }

  @Override
  public void doGet(HttpServletRequest req, HttpServletResponse response) throws IOException {
    DashRequest request;
    try {
      request = new DashRequest(req);
    } catch (DashRequest.DashRequestException e) {
      // TODO(kchodorow): make an error page.
      response.setContentType("text/html");
      response.getWriter().println("Error: " + HtmlEscapers.htmlEscaper().escape(e.getMessage()));
      return;
    }

    BuildData.Builder data = BuildData.newBuilder();
    Query query = new Query(DashRequest.KEY_KIND).setFilter(new FilterPredicate(
        DashRequest.BUILD_ID, FilterOperator.EQUAL, request.getBuildId()));
    PreparedQuery preparedQuery = datastore.prepare(query);
    for (Entity result : preparedQuery.asIterable()) {
      data.mergeFrom(BuildData.parseFrom(
          ((Blob) result.getProperty(DashRequest.BUILD_DATA)).getBytes()));
    }

    VelocityEngine velocityEngine = new VelocityEngine();
    velocityEngine.init();
    Template template = velocityEngine.getTemplate("result.html");
    VelocityContext context = new VelocityContext();

    context.put("build_data", data);

    StringWriter writer = new StringWriter();
    template.merge(context, writer);
    response.setContentType("text/html");
    response.getWriter().println(writer.toString());
  }
}
