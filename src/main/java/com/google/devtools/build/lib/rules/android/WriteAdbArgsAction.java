// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.List;

/**
 * An action that writes the a parameter file to {@code incremental_install.py} based on the command
 * line arguments to {@code blaze mobile-install}.
 */
public class WriteAdbArgsAction extends AbstractFileWriteAction {
  private static final String GUID = "16720416-3c01-4b0a-a543-ead7e563a1ca";

  /**
   * Options of the {@code mobile-install} command pertaining to the way {@code adb} is invoked.
   */
  public static final class Options extends OptionsBase {
    @Option(name = "adb",
        category = "mobile-install",
        defaultValue = "",
        help = "adb binary to use for the 'mobile-install' command. If unspecified, the one in "
            + "the Android SDK specified by the --android_sdk command line option (or the default "
            + "SDK if --android_sdk is not specified) is used.")
    public String adb;

    @Option(name = "adb_arg",
        category = "mobile-install",
        allowMultiple = true,
        defaultValue = "",
        help = "Extra arguments to pass to adb. Usually used to designate a device to install to.")
    public List<String> adbArgs;

    @Option(name = "adb_jobs",
        category = "mobile-install",
        defaultValue = "2",
        help = "The number of instances of adb to use in parallel to update files on the device")
    public int adbJobs;

    @Option(name = "incremental_install_verbosity",
        category = "mobile-install",
        defaultValue = "",
        help = "The verbosity for incremental install. Set to 1 for debug logging.")
    public String incrementalInstallVerbosity;

    @Option(name = "start_app",
        category = "mobile-install",
        defaultValue = "false",
        help = "Whether to start the app after installing it.")
    public boolean startApp;
  }

  public WriteAdbArgsAction(ActionOwner owner, Artifact outputFile) {
    super(owner, ImmutableList.<Artifact>of(), outputFile, false);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(EventHandler eventHandler, Executor executor)
      throws IOException, InterruptedException, ExecException {
    Options options = executor.getOptions().getOptions(Options.class);
    final List<String> args = options.adbArgs;
    final String adb = options.adb;
    final int adbJobs = options.adbJobs;
    final String incrementalInstallVerbosity = options.incrementalInstallVerbosity;
    final boolean startApp = options.startApp;
    final String userHomeDirectory = executor.getContext(
        WriteAdbArgsActionContext.class).getUserHomeDirectory();

    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        PrintStream ps = new PrintStream(out, false, "UTF-8");

        if (!adb.isEmpty()) {
          ps.printf("--adb=%s\n", adb);
        }

        for (String arg : args) {
          ps.printf("--extra_adb_arg=%s\n", arg);
        }

        ps.printf("--adb_jobs=%d\n", adbJobs);

        if (!incrementalInstallVerbosity.isEmpty()) {
          ps.printf("--verbosity=%s\n", incrementalInstallVerbosity);
        }

        ps.printf("--start_app=%s\n", startApp);

        if (userHomeDirectory != null) {
          ps.printf("--user_home_dir=%s\n", userHomeDirectory);
        }

        ps.flush();
      }
    };
  }

  @Override
  public boolean isVolatile() {
    return true;
  }

  @Override
  public boolean executeUnconditionally() {
    // In theory, we only need to re-execute if the --adb_args command line arg changes, but we
    // cannot express this. We also can't put the ADB args in the configuration, because that would
    // mean re-analysis on every change, and then the "build" command would also have this argument,
    // which is not optimal.
    return true;
  }

  @Override
  protected String computeKey() {
    return new Fingerprint()
        .addString(GUID)
        .hexDigestAndReset();
  }
}
