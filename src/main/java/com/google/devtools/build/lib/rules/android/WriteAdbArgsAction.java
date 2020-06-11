// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * An action that writes the a parameter file to {@code incremental_install.py} based on the command
 * line arguments to {@code bazel mobile-install}.
 */
@Immutable // note that it accesses data non-hermetically during the execution phase
public final class WriteAdbArgsAction extends AbstractFileWriteAction {
  private static final String GUID = "16720416-3c01-4b0a-a543-ead7e563a1ca";

  /** Options of the {@code mobile-install} command pertaining to the way {@code adb} is invoked. */
  public static final class Options extends OptionsBase {
    @Option(
      name = "adb",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "adb binary to use for the 'mobile-install' command. If unspecified, the one in "
              + "the Android SDK specified by the --android_sdk command line option (or the "
              + "default SDK if --android_sdk is not specified) is used."
    )
    public String adb;

    @Option(
        name = "adb_arg",
        allowMultiple = true,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
        help = "Extra arguments to pass to adb. Usually used to designate a device to install to.")
    public List<String> adbArgs;

    @Option(
      name = "device",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      help = "The adb device serial number. If not specified, the first device will be used."
    )
    public String device;

    @Option(
      name = "incremental_install_verbosity",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      help = "The verbosity for incremental install. Set to 1 for debug logging."
    )
    public String incrementalInstallVerbosity;

    @Option(
      name = "start",
      converter = StartTypeConverter.class,
      defaultValue = "NO",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "How the app should be started after installing it. Set to WARM to preserve "
              + "and restore application state on incremental installs."
    )
    public StartType start;

    @Option(
      name = "start_app",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "Whether to start the app after installing it.",
      expansion = {"--start=COLD"}
    )
    public Void startApp;

    @Option(
      name = "debug_app",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "Whether to wait for the debugger before starting the app.",
      expansion = {"--start=DEBUG"}
    )
    public Void debugApp;
  }

  public WriteAdbArgsAction(ActionOwner owner, Artifact outputFile) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), outputFile, false);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws IOException, InterruptedException, ExecException {
    Options options = ctx.getOptions().getOptions(Options.class);
    final List<String> args = new ArrayList<>(options.adbArgs);
    final String adb = options.adb;
    final String device = options.device;
    final String incrementalInstallVerbosity = options.incrementalInstallVerbosity;
    final StartType start = options.start;
    final String userHomeDirectory = ctx.getClientEnv().get("HOME");

    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        PrintStream ps = new PrintStream(out, false, "UTF-8");

        if (!adb.isEmpty()) {
          ps.printf("--adb=%s\n", adb);
        }

        if (!device.isEmpty()) {
          args.add("-s");
          args.add(device);
        }

        for (String arg : args) {
          ps.printf("--extra_adb_arg=%s\n", arg);
        }

        if (!incrementalInstallVerbosity.isEmpty()) {
          ps.printf("--verbosity=%s\n", incrementalInstallVerbosity);
        }

        ps.printf("--start=%s\n", start.name().toLowerCase());

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
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
  }

  /** Specifies how the app should be started/stopped. */
  public enum StartType {
    /** The app will not be restarted after install. */
    NO,
    /** The app will be restarted from a clean state after install. */
    COLD,
    /**
     * The app will save its state before installing, and be restored from that state after
     * installing.
     */
    WARM,
    /** The app will wait for debugger to attach before restarting from clean state after install */
    DEBUG
  }

  /** Converter for the --start option. */
  public static class StartTypeConverter extends EnumConverter<StartType> {
    public StartTypeConverter() {
      super(StartType.class, "start type");
    }
  }
}
