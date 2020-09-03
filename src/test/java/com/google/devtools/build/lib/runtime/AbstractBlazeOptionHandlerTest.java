// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.TestOptions;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Abstract class for setting up tests that make use of {@link BlazeOptionHandler}. */
@RunWith(JUnit4.class)
public abstract class AbstractBlazeOptionHandlerTest {

  protected final Scratch scratch = new Scratch();
  protected final StoredEventHandler eventHandler = new StoredEventHandler();
  protected OptionsParser parser;
  protected BlazeRuntime runtime;
  protected BlazeOptionHandler optionHandler;

  @Before
  public void initStuff() throws Exception {
    parser =
        OptionsParser.builder()
            .optionsClasses(TestOptions.class, CommonCommandOptions.class, ClientOptions.class)
            .allowResidue(true)
            .build();
    String productName = TestConstants.PRODUCT_NAME;
    ServerDirectories serverDirectories =
        new ServerDirectories(
            scratch.dir("install_base"), scratch.dir("output_base"), scratch.dir("user_root"));
    this.runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setServerDirectories(serverDirectories)
            .setProductName(productName)
            .setStartupOptionsProvider(
                OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build())
            .addBlazeModule(new BazelRulesModule())
            .build();
    this.runtime.overrideCommands(ImmutableList.of(new C0Command()));

    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories,
            scratch.dir("workspace"),
            /* defaultSystemJavabase= */ null,
            productName);
    runtime.initWorkspace(directories, /*binTools=*/ null);

    optionHandler =
        new BlazeOptionHandler(
            runtime,
            runtime.getWorkspace(),
            new C0Command(),
            C0Command.class.getAnnotation(Command.class),
            parser,
            InvocationPolicy.getDefaultInstance());
  }

  /** Custom command for testing. */
  @Command(
      name = "c0",
      shortDescription = "c0 desc",
      help = "c0 help",
      options = {TestOptions.class})
  protected static class C0Command implements BlazeCommand {
    @Override
    public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void editOptions(OptionsParser optionsParser) {}
  }
}
