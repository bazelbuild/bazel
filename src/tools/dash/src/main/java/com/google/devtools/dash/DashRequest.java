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
import com.google.appengine.api.datastore.Entity;
import com.google.common.io.ByteStreams;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.servlet.http.HttpServletRequest;

/**
 * Parent class for dash-related servlets.
 */
class DashRequest {
  public static final String KEY_KIND = "build";

  public static final String BUILD_ID = "build_id";
  public static final String PAGE_NAME = "page_name";
  public static final String BUILD_DATA = "build_data";

  // URI is something like "/result/d2c64e09-df4e-461d-869e-33f014488655".
  private static final Pattern URI_REGEX = Pattern.compile(
      "/(\\w+)/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})");

  private final String pageName;
  private final String buildId;
  private final Blob blob;

  DashRequest(HttpServletRequest request) throws DashRequestException {
    Matcher matcher = URI_REGEX.matcher(request.getRequestURI());
    if (matcher.find()) {
      pageName = matcher.group(1);
      buildId = matcher.group(2);
    } else {
      throw new DashRequestException("Invalid URI pattern: " + request.getRequestURI());
    }
    try {
      // Requests are capped at 32MB (see
      // https://cloud.google.com/appengine/docs/quotas?csw=1#Requests).
      blob = new Blob(ByteStreams.toByteArray(request.getInputStream()));
    } catch (IOException e) {
      throw new DashRequestException("Could not read request body: " + e.getMessage());
    }
  }

  public String getBuildId() {
    return buildId;
  }

  public Entity getEntity() {
    Entity entity = new Entity(DashRequest.KEY_KIND);
    entity.setProperty(BUILD_ID, buildId);
    entity.setProperty(PAGE_NAME, pageName);
    entity.setProperty(BUILD_DATA, blob);
    return entity;
  }

  static class DashRequestException extends Exception {
    public DashRequestException(String message) {
      super(message);
    }
  }
}
