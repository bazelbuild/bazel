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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import java.util.Set;

/**
 * Holds information collected for .o bitcode files coming from a ThinLTO C(++) compilation under
 * our control. Currently maps each bitcode file to the corresponding minimized bitcode file that
 * can be used for the LTO indexing step.
 */
public class LtoCompilationContext {
  private final ImmutableMap<Artifact, BitcodeInfo> ltoBitcodeFiles;

  public LtoCompilationContext(ImmutableMap<Artifact, BitcodeInfo> ltoBitcodeFiles) {
    this.ltoBitcodeFiles = ltoBitcodeFiles;
  }

  public LtoCompilationContext(LtoCompilationContext other) {
    this.ltoBitcodeFiles = ImmutableMap.copyOf(other.ltoBitcodeFiles);
  }

  /**
   * Class to hold information for a bitcode file produced by the compile action needed by the LTO
   * indexing and backend actions.
   */
  public static class BitcodeInfo {
    private final Artifact minimizedBitcode;

    public BitcodeInfo(Artifact minimizedBitcode) {
      this.minimizedBitcode = minimizedBitcode;
    }

    /** The minimized bitcode file produced by the compile and used by LTO indexing. */
    public Artifact getMinimizedBitcode() {
      return minimizedBitcode;
    }
  }

  /** Builder for LtoCompilationContext. */
  public static class Builder {
    private final ImmutableMap.Builder<Artifact, BitcodeInfo> ltoBitcodeFiles =
        ImmutableMap.builder();

    public Builder() {}

    public LtoCompilationContext build() {
      return new LtoCompilationContext(ltoBitcodeFiles.build());
    }

    /** Adds a bitcode file with the corresponding minimized bitcode file. */
    public void addBitcodeFile(Artifact fullBitcode, Artifact minimizedBitcode) {
      ltoBitcodeFiles.put(fullBitcode, new BitcodeInfo(minimizedBitcode));
    }

    /** Adds in all bitcode files and associated info from another LtoCompilationContext object. */
    public void addAll(LtoCompilationContext ltoCompilationContext) {
      this.ltoBitcodeFiles.putAll(ltoCompilationContext.ltoBitcodeFiles);
    }
  }

  /** Whether there is an entry for the given bitcode file. */
  public boolean containsBitcodeFile(Artifact fullBitcode) {
    return ltoBitcodeFiles.containsKey(fullBitcode);
  }

  /**
   * Gets the minimized bitcode corresponding to the full bitcode file, or returns full bitcode if
   * it doesn't exist.
   */
  public Artifact getMinimizedBitcodeOrSelf(Artifact fullBitcode) {
    if (!containsBitcodeFile(fullBitcode)) {
      return fullBitcode;
    }
    return ltoBitcodeFiles.get(fullBitcode).getMinimizedBitcode();
  }

  /** Whether the map of bitcode files is empty. */
  public boolean isEmpty() {
    return ltoBitcodeFiles.isEmpty();
  }

  /** The set of bitcode files recorded in the map. */
  public Set<Artifact> getBitcodeFiles() {
    return ltoBitcodeFiles.keySet();
  }
}
