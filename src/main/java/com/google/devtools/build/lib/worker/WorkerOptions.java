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
package com.google.devtools.build.lib.worker;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;

/**
 * Options related to worker processes.
 */
public class WorkerOptions extends OptionsBase {
  public static final WorkerOptions DEFAULTS = Options.getDefaults(WorkerOptions.class);

  @Option(name = "worker_max_instances",
      defaultValue = "4",
      category = "strategy",
      help = "How many instances of a worker process (like the persistent Java compiler) may be "
          + "launched if you use the 'worker' strategy.")
  public int workerMaxInstances;

  @Option(name = "experimental_persistent_javac",
      defaultValue = "null",
      category = "undocumented",
      help = "Enable the experimental persistent Java compiler.",
      expansion = {"--strategy=Javac=worker", "--strategy=JavaIjar=local"})
  public Void experimentalPersistentJavac;

  @Option(
    name = "worker_max_changed_files",
    defaultValue = "0",
    category = "strategy",
    help =
        "Don't run local worker if more files than this were changed. 0 means always use "
            + "workers."
  )
  public int workerMaxChangedFiles;
}
