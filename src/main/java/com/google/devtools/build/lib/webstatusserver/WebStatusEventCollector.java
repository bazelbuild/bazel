// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.webstatusserver;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.rules.test.TestResult;
import com.google.devtools.build.lib.runtime.CommandCompleteEvent;
import com.google.devtools.build.lib.runtime.CommandStartEvent;
import com.google.devtools.build.lib.runtime.TestSummary;
import com.google.devtools.build.lib.syntax.Label;

import java.util.logging.Logger;

/**
 * This class monitors the build progress, collects events and preprocesses them for use by
 * frontend.
 * 
 */
public class WebStatusEventCollector {
  private static final Logger LOG =
      Logger.getLogger(WebStatusEventCollector.class.getCanonicalName());
  private final EventBus eventBus;
  private final Reporter reporter;
  private final int port;
  private WebStatusBuildLog currentBuild;
  private WebStatusServerModule serverModule;

  public WebStatusEventCollector(EventBus eventBus, Reporter reporter,
      WebStatusServerModule webStatusServerModule) {
    this.eventBus = eventBus;
    this.eventBus.register(this);
    this.reporter = reporter;
    this.port = webStatusServerModule.getPort();
    this.serverModule = webStatusServerModule;
    LOG.info("Created new status collector");
  }

  @Subscribe
  public void buildStarted(BuildStartingEvent startingEvent) {
    BuildRequest request = startingEvent.getRequest();
    BlazeVersionInfo versionInfo = BlazeVersionInfo.instance();
    currentBuild.addStartTime(request.getStartTime());
    currentBuild.addTargetList(request.getTargets());
    currentBuild
        .addInfo("version", versionInfo)
        .addInfo("commandName", request.getCommandName())
        .addInfo("outputFs", startingEvent.getOutputFileSystem())
        .addInfo("symlinkPrefix", request.getSymlinkPrefix())
        .addInfo("optionsDescription", request.getOptionsDescription())
        .addInfo("targets", request.getTargets())
        .addInfo("viewOptions", request.getViewOptions());
  }

  @Subscribe
  @SuppressWarnings("unused")
  public void commandComplete(CommandCompleteEvent completeEvent) {
    currentBuild.addInfo("endTime", completeEvent.getEventTimeInEpochTime());
    currentBuild.finish();
  }

  @Subscribe
  @SuppressWarnings("unused")
  public void commandStarted(CommandStartEvent event) {
    this.currentBuild = new WebStatusBuildLog(event.getCommandId());
    this.serverModule.commandStarted();
    String webStatusServerUrl = "http://localhost:" + port;
    this.reporter.handle(Event.info("Status page: " + webStatusServerUrl + "/tests/"
        + this.currentBuild.getCommandId() + " (alternative link: " + webStatusServerUrl
        + WebStatusServerModule.LAST_TEST_URI + " )"));
  }

  @Subscribe
  public void doneTestFiltering(TestFilteringCompleteEvent event) {
    if (event.getTestTargets() != null) {
      Builder<Label> builder = ImmutableList.builder();
      for (ConfiguredTarget target : event.getTestTargets()) {
        builder.add(target.getLabel());
      }
      doneTestFiltering(builder.build());
    }
  }

  @VisibleForTesting
  public void doneTestFiltering(Iterable<Label> testLabels) {
    for (Label label : testLabels) {
      currentBuild.addTestTarget(label);
    }
  }

  @Subscribe
  public void testTargetComplete(TestSummary summary) {
    currentBuild.addTestSummary(summary.getTarget().getLabel(), summary.getStatus(),
        summary.getTestTimes(), summary.isCached());
  }

  @Subscribe
  public void testTargetResult(TestResult result) {
    currentBuild.addTestResult(result.getTestAction().getOwner().getLabel(),
        result.getData().getTestCase(), result.getShardNum());
  }

  @Subscribe
  public void targetComplete(TargetCompleteEvent event) {
    // TODO(bazel-team): would getting more details about failure be useful?
    currentBuild.addTargetBuilt(event.getTarget().getTarget().getLabel(), !event.failed());
  }

  public WebStatusBuildLog getBuildLog() {
    return this.currentBuild;
  }
}
