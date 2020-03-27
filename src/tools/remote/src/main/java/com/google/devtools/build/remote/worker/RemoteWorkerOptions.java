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

package com.google.devtools.build.remote.worker;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.logging.Logger;

/** Options for remote worker. */
public class RemoteWorkerOptions extends OptionsBase {
  private static final Logger logger = Logger.getLogger(RemoteWorkerOptions.class.getName());

  @Option(
    name = "listen_port",
    defaultValue = "8080",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Listening port for the netty server."
  )
  public int listenPort;

  @Option(
    name = "work_path",
    defaultValue = "null",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "A directory for the build worker to do work."
  )
  public String workPath;

  @Option(
    name = "cas_path",
    defaultValue = "null",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "A directory for the build worker to store it's files in. If left unset, and if no "
        + "other store is set, the worker falls back to an in-memory store."
  )
  public String casPath;

  @Option(
    name = "debug",
    defaultValue = "false",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Turn this on for debugging remote job failures. There will be extra messages and the "
            + "work directory will be preserved in the case of failure."
  )
  public boolean debug;

  @Option(
    name = "pid_file",
    defaultValue = "null",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "File for writing the process id for this worker when it is fully started."
  )
  public String pidFile;

  @Option(
    name = "sandboxing",
    defaultValue = "false",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "If supported on this platform, use sandboxing for increased hermeticity."
  )
  public boolean sandboxing;

  @Option(
      name = "sandboxing_writable_path",
      defaultValue = "null",
      category = "build_worker",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help = "When using sandboxing, allow running actions to write to this path.")
  public List<String> sandboxingWritablePaths;

  @Option(
      name = "sandboxing_tmpfs_dir",
      defaultValue = "null",
      category = "build_worker",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help = "When using sandboxing, mount an empty tmpfs onto this path for each running action.")
  public List<String> sandboxingTmpfsDirs;

  @Option(
    name = "sandboxing_block_network",
    defaultValue = "false",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "When using sandboxing, block network access for running actions."
  )
  public boolean sandboxingBlockNetwork;

  @Option(
      name = "jobs",
      defaultValue = "auto",
      converter = JobsConverter.class,
      category = "build_worker",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "The maximum number of concurrent jobs to run. Takes "
              + ResourceConverter.FLAG_SYNTAX
              + ". \"auto\" means to use a reasonable value"
              + " derived from the machine's hardware profile (e.g. the number of processors)."
              + " Values less than 1 or above "
              + MAX_JOBS
              + " are not allowed.")
  public int jobs;

  @Option(
      name = "http_listen_port",
      defaultValue = "0",
      category = "build_worker",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Starts an embedded HTTP REST server on the given port. The server will simply store PUT "
              + "requests in memory and return them again on GET requests. This is useful for "
              + "testing only.")
  public int httpListenPort;

  @Option(
      name = "tls_certificate",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Specify the TLS server certificate to use.")
  public String tlsCertificate;

  @Option(
      name = "tls_private_key",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Specify the TLS private key to be used.")
  public String tlsPrivateKey;

  private static final int MAX_JOBS = 16384;

  /**
   * Converter for jobs. Takes {@value FLAG_SYNTAX}. Values must be between 1 and {@value MAX_JOBS}.
   * Values higher than {@value MAX_JOBS} will be set to {@value MAX_JOBS}.
   */
  public static class JobsConverter extends ResourceConverter {
    public JobsConverter() {
      super(
          () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage()),
          1,
          MAX_JOBS);
    }

    @Override
    public int checkAndLimit(int value) throws OptionsParsingException {
      if (value < minValue) {
        throw new OptionsParsingException(
            String.format("Value '(%d)' must be at least %d.", value, minValue));
      }
      if (value > maxValue) {
        logger.warning(
            String.format(
                "Flag remoteWorker \"jobs\" ('%d') was set too high. "
                    + "This is a result of passing large values to --local_resources or --jobs. "
                    + "Using '%d' jobs",
                value, maxValue));
        value = maxValue;
      }
      return value;
    }
  }
}
