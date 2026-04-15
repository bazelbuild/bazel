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
package com.google.devtools.build.lib.worker;

import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import java.time.Duration;

/** Options for the example worker itself. */
@OptionsClass
public abstract class ExampleWorkerOptions extends OptionsBase {

  /** Options for the example worker concerning single units of work. */
  @OptionsClass
  public abstract static class ExampleWorkOptions extends OptionsBase {
    @Option(
        name = "output_file",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "",
        help = "Write the output to a file instead of stdout.")
    public abstract String getOutputFile();

    @Option(
        name = "uppercase",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Uppercase the input.")
    public abstract boolean getUppercase();

    @Option(
        name = "write_uuid",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Writes a UUID into the output.")
    public abstract boolean getWriteUUID();

    @Option(
        name = "write_counter",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Writes a counter that increases with each work unit processed into the output.")
    public abstract boolean getWriteCounter();

    @Option(
        name = "print_requests",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Prints out all requests.")
    public abstract boolean getPrintRequests();

    @Option(
        name = "print_inputs",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Writes a list of input files and their digests.")
    public abstract boolean getPrintInputs();

    @Option(
        name = "print_dir_listing",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "",
        help = "Writes a recursive listing of the given directory, not following symlinks.")
    public abstract String getPrintDirListing();

    @Option(
        name = "print_env",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false",
        help = "Prints a list of all environment variables.")
    public abstract boolean getPrintEnv();

    @Option(
        name = "work_time",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = Converters.DurationConverter.class,
        defaultValue = "0",
        help =
            "When the worker receives a work request, it will sleep for this long before "
                + "responding.")
    public abstract Duration getWorkTime();
  }

  @Option(
      name = "persistent_worker",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false")
  public abstract boolean getPersistentWorker();

  @Option(
      name = "exit_after",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0",
      help = "The worker exits after processing this many work units (default: disabled).")
  public abstract int getExitAfter();

  @Option(
      name = "exit_during",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0",
      help = "The worker exits during processing after this many work units (default: disabled).")
  public abstract int getExitDuring();

  @Option(
      name = "poison_after",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0",
      help =
          "Poisons the worker after processing this many work units, so that it returns a "
              + "corrupt response instead of a response protobuf from then on (default: disabled).")
  public abstract int getPoisonAfter();

  @Option(
      name = "hard_poison",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "Instead of writing an error message to stdout, write it to stderr and terminate.")
  public abstract boolean getHardPoison();

  @Option(
      name = "wait_for_cancel",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "Don't send a response until receiving a cancel request.")
  public abstract boolean getWaitForCancel();

  @Option(
      name = "ignored_argument",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "An argument that does nothing, but whose presence can be asserted in a test.")
  public abstract boolean getIgnoredArgument();

  /** Enum converter for --worker_protocol. */
  public static class WorkerProtocolEnumConverter
      extends EnumConverter<ExecutionRequirements.WorkerProtocolFormat> {
    public WorkerProtocolEnumConverter() {
      super(ExecutionRequirements.WorkerProtocolFormat.class, "worker protocol format option");
    }
  }

  @Option(
      name = "worker_protocol",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "proto",
      help = "The protocol (JSON or proto) to use for communication between this worker and Bazel.",
      converter = WorkerProtocolEnumConverter.class)
  public abstract ExecutionRequirements.WorkerProtocolFormat getWorkerProtocol();
}
