// Copyright 2014 Google Inc. All rights reserved.
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

import java.util.LinkedHashSet;
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
  private final ImmutableList<Artifact> temps;
  
  /**
   * All token .h.processed files created when preprocessing or parsing headers.
   */
  private final ImmutableList<Artifact> headerTokenFiles;

  private CcCompilationOutputs(ImmutableList<Artifact> objectFiles,
      ImmutableList<Artifact> picObjectFiles, ImmutableList<Artifact> dwoFiles,
      ImmutableList<Artifact> picDwoFiles, ImmutableList<Artifact> temps,
      ImmutableList<Artifact> headerTokenFiles) {
    this.objectFiles = objectFiles;
    this.picObjectFiles = picObjectFiles;
    this.dwoFiles = dwoFiles;
    this.picDwoFiles = picDwoFiles;
    this.temps = temps;
    this.headerTokenFiles = headerTokenFiles;
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
  public ImmutableList<Artifact> getTemps() {
    return temps;
  }

  /**
   * Returns an unmodifiable view of the .h.processed files.
   */
  public Iterable<Artifact> getHeaderTokenFiles() {
    return headerTokenFiles;
  }

  public static final class Builder {
    private final Set<Artifact> objectFiles = new LinkedHashSet<>();
    private final Set<Artifact> picObjectFiles = new LinkedHashSet<>();
    private final Set<Artifact> dwoFiles = new LinkedHashSet<>();
    private final Set<Artifact> picDwoFiles = new LinkedHashSet<>();
    private final Set<Artifact> temps = new LinkedHashSet<>();
    private final Set<Artifact> headerTokenFiles = new LinkedHashSet<>();

    public CcCompilationOutputs build() {
      return new CcCompilationOutputs(ImmutableList.copyOf(objectFiles),
          ImmutableList.copyOf(picObjectFiles), ImmutableList.copyOf(dwoFiles),
          ImmutableList.copyOf(picDwoFiles), ImmutableList.copyOf(temps),
          ImmutableList.copyOf(headerTokenFiles));
    }

    public Builder merge(CcCompilationOutputs outputs) {
      this.objectFiles.addAll(outputs.objectFiles);
      this.picObjectFiles.addAll(outputs.picObjectFiles);
      this.dwoFiles.addAll(outputs.dwoFiles);
      this.picDwoFiles.addAll(outputs.picDwoFiles);
      this.temps.addAll(outputs.temps);
      this.headerTokenFiles.addAll(outputs.headerTokenFiles);
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
      Iterables.addAll(temps, artifacts);
      return this;
    }
    
    public Builder addHeaderTokenFile(Artifact artifact) {
      headerTokenFiles.add(artifact);
      return this;
    }

    public Builder addHeaderTokenFiles(Iterable<Artifact> artifacts) {
      Iterables.addAll(headerTokenFiles, artifacts);
      return this;
    }
  }
}
