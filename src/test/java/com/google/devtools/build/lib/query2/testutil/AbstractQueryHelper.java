// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.testutil;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.testutil.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;

/** Partial {@link QueryHelper} implementation for settings storage and event handling. */
public abstract class AbstractQueryHelper<T> implements QueryHelper<T> {
  private Reporter reporter;
  private EventCollector eventCollector;

  protected boolean keepGoing;
  protected ImmutableSet<Setting> settings = ImmutableSet.of();
  protected boolean orderedResults = true;
  protected UniverseScope universeScope = UniverseScope.EMPTY;

  protected TargetPattern.Parser mainRepoTargetParser = TargetPattern.defaultParser();

  @Override
  public void setUp() throws Exception {
    eventCollector = new EventCollector(EventKind.ERRORS_AND_WARNINGS);
    reporter = new Reporter(new EventBus(), eventCollector);
  }

  public Reporter getReporter() {
    return reporter;
  }

  @Override
  public void setUniverseScope(String universeScope) {
    this.universeScope =
        UniverseScope.fromUniverseScopeList(
            ImmutableList.copyOf(Arrays.asList(universeScope.split(","))));
  }

  @Override
  public void clearEvents() {
    eventCollector.clear();
  }

  @Override
  public void setOrderedResults(boolean orderedResults) {
    this.orderedResults = orderedResults;
  }

  @Override
  public void setKeepGoing(boolean keepGoing) {
    this.keepGoing = keepGoing;
  }

  @Override
  public boolean isKeepGoing() {
    return keepGoing;
  }

  @Override
  public void setQuerySettings(Setting... settings) {
    this.settings = ImmutableSet.copyOf(settings);
  }

  @Override
  public void assertContainsEvent(String expectedMessage) {
    MoreAsserts.assertContainsEvent(eventCollector, expectedMessage);
  }

  @Override
  public void assertDoesNotContainEvent(String expectedMessage) {
    MoreAsserts.assertDoesNotContainEvent(eventCollector, expectedMessage);
  }

  @Override
  public String getFirstEvent() {
    return eventCollector.iterator().next().getMessage();
  }

  @Override
  public Iterable<Event> getEvents() {
    return eventCollector;
  }

  @Override
  public void addModule(ModuleKey key, String... moduleFileLines) {
    throw new IllegalStateException("Cannot call this on non-bzlmod-enabled query environments.");
  }

  @Override
  public Path getModuleRoot() {
    throw new IllegalStateException("Cannot call this on non-bzlmod-enabled query environments.");
  }

  @Override
  public void setMainRepoTargetParser(RepositoryMapping mapping) {
    this.mainRepoTargetParser =
        new TargetPattern.Parser(PathFragment.EMPTY_FRAGMENT, RepositoryName.MAIN, mapping);
  }

  @Override
  public void maybeHandleDiffs() throws AbruptExitException, InterruptedException {
    // Do nothing.
  }
}
