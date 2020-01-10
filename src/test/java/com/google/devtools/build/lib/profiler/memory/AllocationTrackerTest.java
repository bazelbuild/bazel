// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler.memory;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleFunction;
import com.google.devtools.build.lib.profiler.memory.AllocationTracker.RuleBytes;
import com.google.devtools.build.lib.syntax.Callstack;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.SyntaxError;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.perftools.profiles.ProfileProto.Function;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.Sample;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AllocationTracker}. */
@RunWith(JUnit4.class)
public final class AllocationTrackerTest {

  private AllocationTracker allocationTracker;

  // makeExpr returns an expression located at the specified file and line.
  private static Expression makeExpr(String file, int line) throws SyntaxError {
    ParserInput input =
        ParserInput.create(
            new String(new char[line - 1]).replace('\u0000', '\n') + "None",
            PathFragment.create(file));
    return Expression.parse(input);
  }

  private static class TestFunction implements StarlarkCallable {
    private final String name;
    private final Location location;

    TestFunction(String file, String name, int line) {
      this.name = name;
      this.location = location(file, line);
    }

    @Override
    public String getName() {
      return name;
    }

    @Override
    public Location getLocation() {
      return location;
    }
  }

  static class TestRuleFunction extends TestFunction implements RuleFunction {

    private final RuleClass ruleClass;

    TestRuleFunction(String file, String name, int line) {
      super(file, name, line);
      this.ruleClass = mock(RuleClass.class);
      when(ruleClass.getName()).thenReturn(name);
      when(ruleClass.getKey()).thenReturn(name);
    }

    @Override
    public RuleClass getRuleClass() {
      return ruleClass;
    }
  }

  @Before
  public void setup() {
    Callstack.setEnabled(true);
    CurrentRuleTracker.setEnabled(true);
    allocationTracker = new AllocationTracker(1, 0);
  }

  @After
  public void tearDown() {
    Callstack.resetStateForTest();
    CurrentRuleTracker.setEnabled(false);
  }

  @Test
  public void testSimpleMemoryProfile() throws Exception {
    Object allocation = new Object();
    Callstack.push(new TestFunction("fileA", "fn", 120));
    Callstack.push(makeExpr("fileA", 10));
    allocationTracker.sampleAllocation(1, "", allocation, 12);
    Callstack.pop();
    Callstack.pop();

    Map<String, RuleBytes> rules = new HashMap<>();
    Map<String, RuleBytes> aspects = new HashMap<>();
    allocationTracker.getRuleMemoryConsumption(rules, aspects);
    assertThat(rules).isEmpty();
    assertThat(aspects).isEmpty();

    Profile profile = allocationTracker.buildMemoryProfile();
    assertThat(profile.getSampleList()).hasSize(1);
    assertThat(sampleToCallstack(profile, profile.getSample(0))).containsExactly("fileA:fn:10");
  }

  @Test
  public void testLongerCallstack() throws Exception {
    Object allocation = new Object();
    Callstack.push(new TestFunction("fileB", "fnB", 120));
    Callstack.push(makeExpr("fileB", 10));
    Callstack.push(makeExpr("fileB", 12));
    Callstack.push(makeExpr("fileB", 14));
    Callstack.push(makeExpr("fileB", 18));
    Callstack.push(new TestFunction("fileA", "fnA", 120));
    Callstack.push(makeExpr("fileA", 10));
    allocationTracker.sampleAllocation(1, "", allocation, 12);
    for (int i = 0; i < 7; ++i) {
      Callstack.pop();
    }

    Profile profile = allocationTracker.buildMemoryProfile();
    assertThat(profile.getSampleList()).hasSize(1);
    assertThat(sampleToCallstack(profile, profile.getSample(0)))
        .containsExactly("fileB:fnB:18", "fileA:fnA:10");
  }

  @Test
  public void testConfiguredTargetsMemoryAllocation() throws Exception {
    RuleClass ruleClass = mock(RuleClass.class);
    when(ruleClass.getName()).thenReturn("rule");
    when(ruleClass.getKey()).thenReturn("rule");
    CurrentRuleTracker.beginConfiguredTarget(ruleClass);
    Object ruleAllocation0 = new Object();
    Object ruleAllocation1 = new Object();
    allocationTracker.sampleAllocation(1, "", ruleAllocation0, 10);
    allocationTracker.sampleAllocation(1, "", ruleAllocation1, 20);
    CurrentRuleTracker.endConfiguredTarget();

    CurrentRuleTracker.beginConfiguredAspect(() -> "aspect");
    Object aspectAllocation = new Object();
    allocationTracker.sampleAllocation(1, "", aspectAllocation, 12);
    CurrentRuleTracker.endConfiguredAspect();

    Map<String, RuleBytes> rules = new HashMap<>();
    Map<String, RuleBytes> aspects = new HashMap<>();
    allocationTracker.getRuleMemoryConsumption(rules, aspects);
    assertThat(rules).containsExactly("rule", new RuleBytes("rule").addBytes(30L));
    assertThat(aspects).containsExactly("aspect", new RuleBytes("aspect").addBytes(12L));

    Profile profile = allocationTracker.buildMemoryProfile();
    assertThat(profile.getSampleList()).isEmpty(); // No callstacks
  }

  @Test
  public void testLoadingPhaseRuleAllocations() throws Exception {
    Object allocation = new Object();
    Callstack.push(new TestFunction("fileB", "fnB", 120));
    Callstack.push(makeExpr("fileB", 18));
    Callstack.push(new TestFunction("fileA", "fnA", 120));
    Callstack.push(makeExpr("fileA", 10));
    Callstack.push(new TestRuleFunction("<native>", "proto_library", -1));
    allocationTracker.sampleAllocation(1, "", allocation, 128);
    for (int i = 0; i < 5; ++i) {
      Callstack.pop();
    }

    Map<String, RuleBytes> rules = new HashMap<>();
    Map<String, RuleBytes> aspects = new HashMap<>();
    allocationTracker.getRuleMemoryConsumption(rules, aspects);
    assertThat(rules)
        .containsExactly("proto_library", new RuleBytes("proto_library").addBytes(128L));
  }

  /** Formats a callstack as (file):(method name):(line) */
  private List<String> sampleToCallstack(Profile profile, Sample sample) {
    List<String> result = new ArrayList<>();
    for (long locationId : sample.getLocationIdList()) {
      com.google.perftools.profiles.ProfileProto.Location location =
          profile.getLocation((int) locationId - 1);
      assertThat(location.getLineList()).hasSize(1);
      long functionId = location.getLine(0).getFunctionId();
      long line = location.getLine(0).getLine();
      Function function = profile.getFunction((int) functionId - 1);
      long fileId = function.getFilename();
      long methodId = function.getName();
      String file = profile.getStringTable((int) fileId);
      String method = profile.getStringTable((int) methodId);
      result.add(String.format("%s:%s:%d", file, method, line));
    }
    return result;
  }

  private static Location location(String path, int line) {
    return Location.fromPathAndStartColumn(
        PathFragment.create(path), 0, 0, new LineAndColumn(line, 0));
  }
}
