// Copyright 2017 The Bazel Authors. All rights reserved.
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


import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;

/** Builder for creating a zip filter action. */
public class ZipFilterBuilder {
  /** Type of compression to apply to output archive. */
  public enum Compression {
    /** Output should be compressed. */
    COMPRESSED,
    /** Output should not be compressed. */
    UNCOMPRESSED,
    /** Compression should not change from input Zip. */
    DONT_CHANGE;
  }

  /** Modes of performing content hash checking during zip filtering. */
  public enum CheckHashMismatchMode {
    NONE,
    WARN,
    ERROR;
  }

  private final RuleContext ruleContext;
  private Artifact inputZip;
  private Artifact outputZip;
  private final ImmutableSet.Builder<Artifact> filterZipsBuilder;
  private final ImmutableSet.Builder<String> filterFileTypesBuilder;
  private final ImmutableSet.Builder<String> explicitFilterBuilder;
  private Compression outputMode = Compression.DONT_CHANGE;
  private CheckHashMismatchMode checkHashMismatch = CheckHashMismatchMode.WARN;

  /** Creates a builder using the configuration of the rule as the action configuration. */
  public ZipFilterBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    filterZipsBuilder = new ImmutableSet.Builder<>();
    filterFileTypesBuilder = new ImmutableSet.Builder<>();
    explicitFilterBuilder = new ImmutableSet.Builder<>();
  }

  /** Sets the Zip file to be filtered. */
  public ZipFilterBuilder setInputZip(Artifact inputZip) {
    this.inputZip = inputZip;
    return this;
  }

  /** Sets the artifact to create with the action. */
  public ZipFilterBuilder setOutputZip(Artifact outputZip) {
    this.outputZip = outputZip;
    return this;
  }

  /**
   * Adds to the Zip files to use as filters. Contents in these files will be omitted from the
   * output.
   */
  public ZipFilterBuilder addFilterZips(Iterable<Artifact> filterZips) {
    this.filterZipsBuilder.addAll(filterZips);
    return this;
  }

  /**
   * Adds to the file types to use as filters. Only contents in the filter Zip files with these
   * extensions will be filtered out.
   */
  public ZipFilterBuilder addFileTypeToFilter(String filterFileType) {
    this.filterFileTypesBuilder.add(filterFileType);
    return this;
  }

  /** Adds filterRegex to the set of filters to always check for and remove. */
  public ZipFilterBuilder addExplicitFilter(String filterRegex) {
    this.explicitFilterBuilder.add(filterRegex);
    return this;
  }

  /** Enable checking of hash mismatches for files with the same name. */
  public ZipFilterBuilder setCheckHashMismatchMode(CheckHashMismatchMode mode) {
    this.checkHashMismatch = mode;
    return this;
  }

  /** Builds the action as configured. */
  public void build() {
    ImmutableSet<Artifact> filterZips = filterZipsBuilder.build();
    ImmutableSet<String> filterFileTypes = filterFileTypesBuilder.build();
    ImmutableSet<String> explicitFilters = explicitFilterBuilder.build();

    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--inputZip", inputZip);
    args.addExecPath("--outputZip", outputZip);
    if (!filterZips.isEmpty()) {
      args.addExecPaths("--filterZips", VectorArg.join(",").each(filterZips));
    }
    if (!filterFileTypes.isEmpty()) {
      args.addAll("--filterTypes", VectorArg.join(",").each(filterFileTypes));
    }
    if (!explicitFilters.isEmpty()) {
      args.addAll("--explicitFilters", VectorArg.join(",").each(explicitFilters));
    }
    switch (checkHashMismatch) {
      case WARN:
        args.add("--checkHashMismatch").add("WARN");
        break;
      case ERROR:
        args.add("--checkHashMismatch").add("ERROR");
        break;
      case NONE:
        args.add("--checkHashMismatch").add("IGNORE");
        break;
    }
    args.add("--outputMode");
    switch (outputMode) {
      case COMPRESSED:
        args.add("FORCE_DEFLATE");
        break;
      case UNCOMPRESSED:
        args.add("FORCE_STORED");
        break;
      case DONT_CHANGE:
      default:
        args.add("DONT_CARE");
        break;
    }

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(inputZip)
            .addInputs(filterZips)
            .addOutput(outputZip)
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$zip_filter", TransitionMode.HOST))
            .addCommandLine(args.build())
            .setProgressMessage("Filtering Zip %s", inputZip.prettyPrint())
            .setMnemonic("ZipFilter")
            .build(ruleContext));
  }
}
