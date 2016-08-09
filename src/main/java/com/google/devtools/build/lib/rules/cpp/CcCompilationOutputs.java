// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A structured representation of the compilation outputs of a C++ rule.
 */
public class CcCompilationOutputs {
  /**
   * All .o files built by the target.
   */
  private final ImmutableList<Artifact> objectFiles;

  /**
   * All .pic.o files built by the target.
   */
  private final ImmutableList<Artifact> picObjectFiles;

  /**
   * All .o files coming from a C(++) compilation under our control.
   */
  private final ImmutableList<Artifact> ltoBitcodeFiles;

  /**
   * All .dwo files built by the target, corresponding to .o outputs.
   */
  private final ImmutableList<Artifact> dwoFiles;

  /**
   * All .pic.dwo files built by the target, corresponding to .pic.o outputs.
   */
  private final ImmutableList<Artifact> picDwoFiles;

  /**
   * All artifacts that are created if "--save_temps" is true.
   */
  private final NestedSet<Artifact> temps;

  /**
   * All token .h.processed files created when preprocessing or parsing headers.
   */
  private final ImmutableList<Artifact> headerTokenFiles;

  private final List<IncludeScannable> lipoScannables;

  private CcCompilationOutputs(
      ImmutableList<Artifact> objectFiles,
      ImmutableList<Artifact> picObjectFiles,
      ImmutableList<Artifact> ltoBitcodeFiles,

      ImmutableList<Artifact> dwoFiles,
      ImmutableList<Artifact> picDwoFiles,
      NestedSet<Artifact> temps,
      ImmutableList<Artifact> headerTokenFiles,
      ImmutableList<IncludeScannable> lipoScannables) {
    this.objectFiles = objectFiles;
    this.picObjectFiles = picObjectFiles;
    this.ltoBitcodeFiles = ltoBitcodeFiles;
    this.dwoFiles = dwoFiles;
    this.picDwoFiles = picDwoFiles;
    this.temps = temps;
    this.headerTokenFiles = headerTokenFiles;
    this.lipoScannables = lipoScannables;
  }

  /**
   * Returns whether this set of outputs has any object or .pic object files.
   */
  public boolean isEmpty() {
    return picObjectFiles.isEmpty() && objectFiles.isEmpty();
  }

  /**
   * Returns an unmodifiable view of the .o or .pic.o files set.
   *
   * @param usePic whether to return .pic.o files
   */
  public ImmutableList<Artifact> getObjectFiles(boolean usePic) {
    return usePic ? picObjectFiles : objectFiles;
  }

  /**
   * Returns unmodifiable view of object files resulting from compilation.
   */
  public ImmutableList<Artifact> getLtoBitcodeFiles() {
    return ltoBitcodeFiles;
  }

  /**
   * Returns an unmodifiable view of the .dwo files set.
   */
  public ImmutableList<Artifact> getDwoFiles() {
    return dwoFiles;
  }

  /**
   * Returns an unmodifiable view of the .pic.dwo files set.
   */
  public ImmutableList<Artifact> getPicDwoFiles() {
    return picDwoFiles;
  }

  /**
   * Returns an unmodifiable view of the temp files set.
   */
  public NestedSet<Artifact> getTemps() {
    return temps;
  }

  /**
   * Returns an unmodifiable view of the .h.processed files.
   */
  public Iterable<Artifact> getHeaderTokenFiles() {
    return headerTokenFiles;
  }

  /**
   * Returns the {@link IncludeScannable} objects this C++ compile action contributes to a
   * LIPO context collector.
   */
  public List<IncludeScannable> getLipoScannables() {
    return lipoScannables;
  }
  
  /**
   * Returns the output files that are considered "compiled" by this C++ compile action.
   */
  NestedSet<Artifact> getFilesToCompile(
      boolean isLipoContextCollector, boolean parseHeaders, boolean usePic) {
    if (isLipoContextCollector) {
      return NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
    }
    NestedSetBuilder<Artifact> files = NestedSetBuilder.stableOrder();
    files.addAll(getObjectFiles(usePic));
    if (parseHeaders) {
      files.addAll(getHeaderTokenFiles());
    }
    return files.build();
  }


  public static final class Builder {
    private final Set<Artifact> objectFiles = new LinkedHashSet<>();
    private final Set<Artifact> picObjectFiles = new LinkedHashSet<>();
    private final Set<Artifact> ltoBitcodeFiles = new LinkedHashSet<>();
    private final Set<Artifact> dwoFiles = new LinkedHashSet<>();
    private final Set<Artifact> picDwoFiles = new LinkedHashSet<>();
    private final NestedSetBuilder<Artifact> temps = NestedSetBuilder.stableOrder();
    private final Set<Artifact> headerTokenFiles = new LinkedHashSet<>();
    private final List<IncludeScannable> lipoScannables = new ArrayList<>();

    public CcCompilationOutputs build() {
      return new CcCompilationOutputs(
          ImmutableList.copyOf(objectFiles),
          ImmutableList.copyOf(picObjectFiles),
          ImmutableList.copyOf(ltoBitcodeFiles),
          ImmutableList.copyOf(dwoFiles),
          ImmutableList.copyOf(picDwoFiles),
          temps.build(),
          ImmutableList.copyOf(headerTokenFiles),
          ImmutableList.copyOf(lipoScannables));
    }

    public Builder merge(CcCompilationOutputs outputs) {
      this.objectFiles.addAll(outputs.objectFiles);
      this.picObjectFiles.addAll(outputs.picObjectFiles);
      this.dwoFiles.addAll(outputs.dwoFiles);
      this.picDwoFiles.addAll(outputs.picDwoFiles);
      this.temps.addTransitive(outputs.temps);
      this.headerTokenFiles.addAll(outputs.headerTokenFiles);
      this.lipoScannables.addAll(outputs.lipoScannables);
      return this;
    }

    /**
     * Adds an .o file.
     */
    public Builder addObjectFile(Artifact artifact) {
      objectFiles.add(artifact);
      return this;
    }

    public Builder addObjectFiles(Iterable<Artifact> artifacts) {
      Iterables.addAll(objectFiles, artifacts);
      return this;
    }

    /**
     * Adds a .pic.o file.
     */
    public Builder addPicObjectFile(Artifact artifact) {
      picObjectFiles.add(artifact);
      return this;
    }

    public Builder addLTOBitcodeFile(Artifact a) {
      ltoBitcodeFiles.add(a);
      return this;
    }

    public Builder addLTOBitcodeFile(Iterable<Artifact> artifacts) {
      Iterables.addAll(ltoBitcodeFiles, artifacts);
      return this;
    }

    public Builder addPicObjectFiles(Iterable<Artifact> artifacts) {
      Iterables.addAll(picObjectFiles, artifacts);
      return this;
    }

    public Builder addDwoFile(Artifact artifact) {
      dwoFiles.add(artifact);
      return this;
    }

    public Builder addPicDwoFile(Artifact artifact) {
      picDwoFiles.add(artifact);
      return this;
    }

    /**
     * Adds temp files.
     */
    public Builder addTemps(Iterable<Artifact> artifacts) {
      temps.addAll(artifacts);
      return this;
    }

    public Builder addHeaderTokenFile(Artifact artifact) {
      headerTokenFiles.add(artifact);
      return this;
    }

    /**
     * Adds an {@link IncludeScannable} that this compilation output object contributes to a
     * LIPO context collector.
     */
    public Builder addLipoScannable(IncludeScannable scannable) {
      lipoScannables.add(scannable);
      return this;
    }
  }
}
