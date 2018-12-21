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
// limitations under the License.
package com.google.devtools.build.lib.includescanning;

import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Command options specific to include scanning.
 */
public class IncludeScanningOptions extends OptionsBase {

  /**
   * Converter for scanning parallelism threads: Takes {@value #FLAG_SYNTAX} 0 disables scanning
   * parallelism.
   */
  public static class ParallelismConverter extends ResourceConverter {
    public ParallelismConverter() throws OptionsParsingException {
      super(
          /* autoSupplier= */ () ->
              (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage()),
          /* minValue= */ 0,
          /* maxValue= */ Integer.MAX_VALUE);
    }
  }

  @Option(
    name = "experimental_inmemory_dotincludes_files",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.EXECUTION,
      OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS
    },
    defaultValue = "false",
    help =
        "If enabled, searching for '#include' lines in generated header files will not "
            + "touch local disk. This makes include scanning of C++ files less disk-intensive."
  )
  public boolean inMemoryIncludesFiles;

  @Option(
      name = "experimental_remote_include_extraction_size_threshold",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
          OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
          OptionEffectTag.EXECUTION,
          OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS
      },
      defaultValue = "1000000",
      help = "Run remotable C++ include extraction remotely if the file size in bytes exceeds this."
  )
  public int experimentalRemoteExtractionThreshold;

  @Option(
    name = "experimental_skyframe_include_scanning",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.EXECUTION,
      OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS
    },
    defaultValue = "false",
    deprecationWarning = "No longer active: is a no-op",
    help = "Deprecated, has no effect."
  )
  public boolean skyframeIncludeScanning;

  @Option(
      name = "experimental_include_scanning_parallelism",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.EXECUTION,
        OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS
      },
      defaultValue = "80",
      converter = ParallelismConverter.class,
      help =
          "Configures the size of the thread pool used for include scanning. Takes "
              + ResourceConverter.FLAG_SYNTAX
              + ". 0 means to disable parallelism and to just rely on the build graph parallelism "
              + "for concurrency. "
              + " \"auto\" means to use a reasonable value derived from the machine's hardware"
              + " profile (e.g. the number of processors).")
  public int includeScanningParallelism;
}
