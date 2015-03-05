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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * Stores information about one build command. The data is stored in JSON so that it can be
 * can be easily fed to frontend.
 *
 * <p> The information is grouped into following structures:
 * <ul>
 * <li> {@link #commandInfo} contain information about the build known when it starts but before
 *      anything is actually compiled/run
 * <li> {@link #testCases} contain detailed information about each test case ran, for now they're
 *
 * </ul>
 */
public class WebStatusBuildLog {
  private Gson gson = new Gson();
  private boolean complete = false;
  private static final Logger LOG =
      Logger.getLogger(WebStatusEventCollector.class.getCanonicalName());
  private Map<String, JsonElement> commandInfo = new HashMap<>();
  private Map<String, JsonObject> testCases = new HashMap<>();
  private long startTime;
  private ImmutableList<String> targetList;
  private UUID commandId;

  public WebStatusBuildLog(UUID commandId) {
    this.commandId = commandId;
  }

  public WebStatusBuildLog addInfo(String key, Object value) {
    commandInfo.put(key, gson.toJsonTree(value));
    return this;
  }

  public void addStartTime(long startTime) {
    this.startTime = startTime;
  }

  public void addTargetList(List<String> targets) {
    this.targetList = ImmutableList.copyOf(targets);
  }

  public void finish() {
    commandInfo = ImmutableMap.copyOf(commandInfo);
    complete = true;
  }

  public Map<String, JsonElement> getCommandInfo() {
    return commandInfo;
  }

  public ImmutableMap<String, JsonObject> getTestCases() {
    // TODO(bazel-team): not really immutable, since one can do addProperty on
    // values (unfortunately gson doesn't support immutable JsonObjects)
    return ImmutableMap.copyOf(testCases);
  }

  public boolean finished() {
    return complete;
  }

  public List<String> getTargetList() {
    return targetList;
  }

  public long getStartTime() {
    return startTime;
  }

  public void addTestTarget(Label label) {
    String targetName = label.toShorthandString();
    if (!testCases.containsKey(targetName)) {
      JsonObject summary = createTestCaseEmptyJsonNode(targetName);
      summary.addProperty("finished", false);
      summary.addProperty("status", "started");
      testCases.put(targetName, summary);
    } else {
      // TODO(bazel-team): figure out if there are any situations it can happen
    }
  }

  public void addTestSummary(Label label, BlazeTestStatus status, List<Long> testTimes,
      boolean isCached) {
    JsonObject testCase = testCases.get(label.toShorthandString());
    testCase.addProperty("status", status.toString());
    testCase.add("times", gson.toJsonTree(testTimes));
    testCase.addProperty("cached", isCached);
    testCase.addProperty("finished", true);
  }

  public void addTargetBuilt(Label label, boolean success) {
    if (testCases.containsKey(label.toShorthandString())) {
      if (success) {
        testCases.get(label.toShorthandString()).addProperty("status", "built");
      } else {
        testCases.get(label.toShorthandString()).addProperty("status", "build failure");
      }
    } else {
      LOG.info("Unhandled target: " + label);
    }
  }

  @VisibleForTesting
  static JsonObject createTestCaseEmptyJsonNode(String fullName) {
    JsonObject currentNode = new JsonObject();
    currentNode.addProperty("fullName", fullName);
    currentNode.addProperty("name", "");
    currentNode.addProperty("className", "");
    currentNode.add("results", new JsonObject());
    currentNode.add("times", new JsonObject());
    currentNode.add("children", new JsonObject());
    currentNode.add("failures", new JsonObject());
    currentNode.add("errors", new JsonObject());
    return currentNode;
  }

  private static JsonObject createTestCaseEmptyJsonNode(String fullName, TestCase testCase) {
    JsonObject currentNode = createTestCaseEmptyJsonNode(fullName);
    currentNode.addProperty("name", testCase.getName());
    currentNode.addProperty("className", testCase.getClassName());
    return currentNode;
  }

  private JsonObject mergeTestCases(JsonObject currentNode, String fullName, TestCase testCase,
      int shardNumber) {
    if (currentNode == null) {
      currentNode = createTestCaseEmptyJsonNode(fullName, testCase);
    }

    if (testCase.getRun()) {
      JsonObject results = (JsonObject) currentNode.get("results");
      JsonObject times = (JsonObject) currentNode.get("times");

      if (testCase.hasResult()) {
        results.addProperty(Integer.toString(shardNumber), testCase.getResult());
      }

      if (testCase.hasStatus()) {
        results.addProperty(Integer.toString(shardNumber), testCase.getStatus().toString());
      }

      if (testCase.hasRunDurationMillis()) {
        times.addProperty(Integer.toString(shardNumber), testCase.getRunDurationMillis());
      }
    }
    JsonObject children = (JsonObject) currentNode.get("children");

    for (TestCase child : testCase.getChildList()) {
      String fullChildName = child.getClassName() + "." + child.getName();
      JsonObject childNode = mergeTestCases((JsonObject) children.get(fullChildName), fullChildName,
          child, shardNumber);
      if (!children.has(fullChildName)) {
        children.add(fullChildName, childNode);
      }
    }
    return currentNode;
  }

  public void addTestResult(Label label, TestCase testCase, int shardNumber) {
    String testResultFullName = label.toShorthandString();
    if (!testCases.containsKey(testResultFullName)) {
      testCases.put(testResultFullName, createTestCaseEmptyJsonNode(testResultFullName, testCase));
    }
    mergeTestCases(testCases.get(testResultFullName), testResultFullName, testCase, shardNumber);
  }

  public UUID getCommandId() {
    return commandId;
  }
}
