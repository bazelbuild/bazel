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

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownBlazeServerException;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
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
import java.util.Map;
import java.util.Set;

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

    @Option(name = "stringoption", defaultValue = "[unspecified]")
    public String stringOption;
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

  @Command(
    name = "reportall",
    options = {FooOptions.class},
    shortDescription = "",
    help = ""
  )
  private static class ReportAllCommand implements BlazeCommand {

    @Override
    public ExitCode exec(CommandEnvironment env, OptionsProvider options)
        throws ShutdownBlazeServerException {
      FooOptions fooOptions = options.getOptions(FooOptions.class);
      env.getReporter()
          .getOutErr()
          .printOut("" + fooOptions.numOption + " " + fooOptions.stringOption);
      return ExitCode.SUCCESS;
    }

    @Override
    public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}
  }

  @Command(
    name = "reportallinherited",
    options = {FooOptions.class},
    shortDescription = "",
    help = "",
    inherits = ReportAllCommand.class
  )
  private static class ReportAllInheritedCommand extends ReportAllCommand {
  }


  private final Scratch scratch = new Scratch();
  private final RecordingOutErr outErr = new RecordingOutErr();
  private final ReportNumCommand reportNum = new ReportNumCommand();
  private final ReportAllCommand reportAll = new ReportAllCommand();
  private final ReportAllCommand reportAllInherited = new ReportAllInheritedCommand();
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
            .addBlazeModule(
                new BlazeModule() {
                  @Override
                  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
                    // We must add these options so that the defaults package can be created.
                    builder.addConfigurationOptions(BuildConfiguration.Options.class);
                    // The defaults package asserts that it is not empty, so we provide options.
                    builder.addConfigurationOptions(MockFragmentOptions.class);
                    // The tools repository is needed for createGlobals
                    builder.setToolsRepository(TestConstants.TOOLS_REPOSITORY);
                  }
                })
            .setInvocationPolicy(InvocationPolicyOuterClass.InvocationPolicy.getDefaultInstance())
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

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
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

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
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

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
    String out = outErr.outAsLatin1();
    assertEquals("Specific options should dominate common options", "42", out);
  }

  @Test
  public void testOptionsCombined() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/etc/bazelrc",
            "--default_override=0:common=--stringoption=foo",
            "--rc_source=/home/jrluser/.blazerc",
            "--default_override=1:common=--numoption=99");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum, reportAll);
    List<String> cmdLine = Lists.newArrayList("reportall");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
    String out = outErr.outAsLatin1();
    assertEquals("Options should get accumulated over different rc files", "99 foo", out);
  }

  @Test
  public void testOptionsCombinedWithOverride() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/etc/bazelrc",
            "--default_override=0:common=--stringoption=foo",
            "--default_override=0:common=--numoption=42",
            "--rc_source=/home/jrluser/.blazerc",
            "--default_override=1:common=--numoption=99");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum, reportAll);
    List<String> cmdLine = Lists.newArrayList("reportall");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
    String out = outErr.outAsLatin1();
    assertEquals("The more specific rc-file should override", "99 foo", out);
  }

  @Test
  public void testOptionsCombinedWithOverrideOtherName() throws Exception {
    List<String> blazercOpts =
        ImmutableList.of(
            "--rc_source=/home/jrluser/.blazerc",
            "--default_override=0:common=--stringoption=foo",
            "--default_override=0:common=--numoption=42",
            "--rc_source=/etc/bazelrc",
            "--default_override=1:common=--numoption=99");

    BlazeCommandDispatcher dispatch = new BlazeCommandDispatcher(runtime, reportNum, reportAll);
    List<String> cmdLine = Lists.newArrayList("reportall");
    cmdLine.addAll(blazercOpts);

    dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
    String out = outErr.outAsLatin1();
    assertEquals("The more specific rc-file should override irrespective of name", "99 foo", out);
  }

  @Test
  public void testInheritedOptionsWithSpecificOverride() throws Exception {
    ImmutableList<ImmutableList<String>> blazercOpts =
        ImmutableList.of(
            ImmutableList.of(
                "--rc_source=/doesnt/matter/0/bazelrc",
                "--default_override=0:common=--stringoption=common",
                "--default_override=0:common=--numoption=42"),
            ImmutableList.of(
                "--rc_source=/doesnt/matter/1/bazelrc",
                "--default_override=0:reportall=--stringoption=reportall"),
            ImmutableList.of(
                "--rc_source=/doesnt/matter/2/bazelrc",
                "--default_override=0:reportallinherited=--stringoption=reportallinherited"));

    for (List<ImmutableList<String>> e : Collections2.permutations(blazercOpts)) {
      outErr.reset();
      BlazeCommandDispatcher dispatch =
          new BlazeCommandDispatcher(runtime, reportNum, reportAll, reportAllInherited);
      List<String> cmdLine = Lists.newArrayList("reportallinherited");
      List<String> orderedOpts = ImmutableList.copyOf(Iterables.concat(e));
      cmdLine.addAll(orderedOpts);

      dispatch.exec(cmdLine, LockingMode.ERROR_OUT, "test", outErr);
      String out = outErr.outAsLatin1();
      assertEquals(
          String.format(
              "The more specific option should override, irrespective of source file or order. %s",
              orderedOpts),
          "42 reportallinherited",
          out);
    }
  }

  /** Options class for testing, so that defaults package has some content. */
  public static class MockFragmentOptions extends FragmentOptions {
    public MockFragmentOptions() {}

    @Option(name = "fake_opt", defaultValue = "false")
    public boolean fakeOpt;

    @Override
    public Map<String, Set<Label>> getDefaultsLabels(BuildConfiguration.Options commonOptions) {
      return ImmutableMap.<String, Set<Label>>of(
          "mock_target", ImmutableSet.of(Label.parseAbsoluteUnchecked("//mock:target")));
    }
  }
}
