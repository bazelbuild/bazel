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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.engine.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Arrays;
import java.util.List;

/** Partial {@link QueryHelper} implementation for settings storage and event handling. */
public abstract class AbstractQueryHelper<T> implements QueryHelper<T> {
  protected Reporter reporter;
  protected EventCollector eventCollector;

  protected boolean keepGoing;
  protected ImmutableSet<Setting> settings = ImmutableSet.of();
  protected boolean orderedResults = true;
  protected List<String> universeScope = ImmutableList.of();

  @Override
  public void setUp() throws Exception {
    eventCollector = new EventCollector(EventKind.ERRORS_AND_WARNINGS);
    reporter = new Reporter(new EventBus(), eventCollector);
  }

  @Override
  public void setUniverseScope(String universeScope) {
    this.universeScope = Arrays.asList(universeScope.split(","));
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
}
