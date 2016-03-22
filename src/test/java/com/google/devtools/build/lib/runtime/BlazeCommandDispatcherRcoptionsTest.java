// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownBlazeServerException;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

import java.util.List;

/**
 * Tests the handling of rc-options in {@link BlazeCommandDispatcher}.
 */
@RunWith(JUnit4.class)
public class BlazeCommandDispatcherRcoptionsTest {

  /**
   * Example options to be used by the tests.
   */
  public static class FooOptions extends OptionsBase {
    @Option(name = "numoption", defaultValue = "0")
    public int numOption;
  }

  @Command(
    name = "reportnum",
    options = {FooOptions.class},
    shortDescription = "",
    help = ""
  )
  private static class ReportNumCommand implements BlazeCommand {

    @Override
    public ExitCode exec(CommandEnvironment env, OptionsProvider options)
        throws ShutdownBlazeServerException {
      FooOptions fooOptions = options.getOptions(FooOptions.class);
      env.getReporter().getOutErr().printOut("" + fooOptions.numOption);
      return ExitCode.SUCCESS;
    }

    @Override
    public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}
  }

  private final Scratch scratch = new Scratch();
  private final RecordingOutErr outErr = new RecordingOutErr();
  private final ReportNumCommand reportNum = new ReportNumCommand();
  private BlazeRuntime runtime;

  @Before
  public final void initializeRuntime() throws Exception {
    BlazeDirectories directories =
        new BlazeDirectories(
            scratch.dir("install_base"), scratch.dir("output_base"), scratch.dir("pkg"));
    this.runtime =
        new BlazeRuntime.Builder()
            .setDirectories(directories)
            .setStartupOptionsProvider(
                OptionsParser.newOptionsParser(BlazeServerStartupOptions.class))
            .setConfigurationFactory(
                new ConfigurationFactory(Mockito.mock(ConfigurationCollectionFactory.class)))
            .build();
  }

  @Test
  public void testCommonUsed() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/home/jrluser/.blazerc", "--default_override=0:common=--numoption=99");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum);
    List<String> cmdLine = Lists.newArrayList("reportnum");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, outErr);
    String out = outErr.outAsLatin1();
    assertEquals("Common options should be used", "99", out);
  }

  @Test
  public void testSpecificOptionsWin() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/home/jrluser/.blazerc",
            "--default_override=0:reportnum=--numoption=42",
            "--default_override=0:common=--numoption=99");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum);
    List<String> cmdLine = Lists.newArrayList("reportnum");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, outErr);
    String out = outErr.outAsLatin1();
    assertEquals("Specific options should dominate common options", "42", out);
  }

  @Test
  public void testSpecificOptionsWinOtherOrder() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/home/jrluser/.blazerc",
            "--default_override=0:common=--numoption=99",
            "--default_override=0:reportnum=--numoption=42");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum);
    List<String> cmdLine = Lists.newArrayList("reportnum");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, outErr);
    String out = outErr.outAsLatin1();
    assertEquals("Specific options should dominate common options", "42", out);
  }
}
