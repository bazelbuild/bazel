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
// limitations under the License

package com.google.devtools.build.lib.bazel.debug;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests handling of WorkspaceRuleEvent */
@RunWith(JUnit4.class)
public final class WorkspaceRuleEventTest {
  static class DummyString {
    @Override
    public String toString() {
      return "dummy string";
    }
  }

  static class DummyLocation extends Location {
    public DummyLocation() {
      super(10, 20);
    };

    @Override
    public PathFragment getPath() {
      return null;
    }

    @Override
    public String print() {
      return "location being printed";
    }
  }

  @Before
  public void setUp() {}

  @Test
  public void newExecuteEvent_expectedResult() {
    // Set up arguments, as a combination of String and SkylarkPath
    ArrayList<Object> arguments = new ArrayList<>();
    arguments.add("argument 1");
    arguments.add(new DummyString());

    Map<String, String> commonEnv = ImmutableMap.of("key1", "val1", "key3", "val3");
    Map<String, String> customEnv = ImmutableMap.of("key2", "val2!", "key3", "val3!");

    WorkspaceLogProtos.WorkspaceEvent event =
        WorkspaceRuleEvent.newExecuteEvent(
                arguments,
                2042,
                commonEnv,
                customEnv,
                "outputDir",
                true,
                "my_rule",
                new DummyLocation())
            .getLogEvent();

    List<String> expectedArgs = Arrays.asList("argument 1", "dummy string");

    Map<String, String> expectedEnv =
        ImmutableMap.of(
            "key1", "val1",
            "key2", "val2!",
            "key3", "val3!");

    assertThat(event.getRule()).isEqualTo("my_rule");
    assertThat(event.getLocation()).isEqualTo("location being printed");

    WorkspaceLogProtos.ExecuteEvent executeEvent = event.getExecuteEvent();
    assertThat(executeEvent.getTimeoutSeconds()).isEqualTo(2042);
    assertThat(executeEvent.getQuiet()).isEqualTo(true);
    assertThat(executeEvent.getOutputDirectory()).isEqualTo("outputDir");
    assertThat(executeEvent.getArgumentsList()).isEqualTo(expectedArgs);
    assertThat(executeEvent.getEnvironmentMap()).isEqualTo(expectedEnv);
  }
}
