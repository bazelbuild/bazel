// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.toMap;

import com.google.common.collect.Lists;
import com.google.devtools.build.docgen.builtin.BuiltinProtos;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link HelpCommand}. */
@RunWith(JUnit4.class)
public final class HelpCommandTest extends BuildIntegrationTestCase {
  private BlazeCommandDispatcher dispatcher;
  private RecordingOutErr recordingOutErr;

  @Before
  public void createDispatcher() {
    BlazeRuntime runtime = getRuntime();
    runtime.getCommandMap().put("help", new HelpCommand());
    dispatcher = new BlazeCommandDispatcher(runtime);
  }

  @Before
  public void createRecording() {
    recordingOutErr = new RecordingOutErr();
  }

  private BlazeCommandResult help(String... args) throws InterruptedException {
    List<String> params = Lists.newArrayList("help");
    Collections.addAll(params, args);
    return dispatcher.exec(params, "test", recordingOutErr);
  }

  @Test
  public void wellKnownCallablesInBuiltinSymbolsProto() throws Exception {
    assertThat(help("builtin-symbols-as-proto").isSuccess()).isTrue();
    assertThat(recordingOutErr.errAsLatin1()).isEmpty();
    String base64Proto = recordingOutErr.outAsLatin1();
    byte[] rawProto = Base64.getDecoder().decode(base64Proto);
    var builtins = BuiltinProtos.Builtins.parseFrom(rawProto);

    assertThat(
            builtins.getGlobalList().stream()
                .filter(BuiltinProtos.Value::hasCallable)
                .filter(global -> !global.getCallable().getParamList().isEmpty())
                .collect(toMap(BuiltinProtos.Value::getName, BuiltinProtos.Value::getApiContext)))
        .containsAtLeast(
            "range", BuiltinProtos.ApiContext.ALL,
            "glob", BuiltinProtos.ApiContext.BUILD,
            "DefaultInfo", BuiltinProtos.ApiContext.BZL);
  }
}
