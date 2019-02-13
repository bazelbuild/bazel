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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import java.util.Objects;
import java.util.Set;

/**
 * Holds information collected for .o bitcode files coming from a ThinLTO C(++) compilation under
 * our control. Specifically, maps each bitcode file to the corresponding minimized bitcode file
 * that can be used for the LTO indexing step, as well as to compile flags applying to that
 * compilation that should also be applied to the LTO backend compilation invocation.
 */
public final class LtoCompilationContext {
  static final LtoCompilationContext EMPTY = new LtoCompilationContext(ImmutableMap.of());

  private final ImmutableMap<Artifact, BitcodeInfo> ltoBitcodeFiles;

  private LtoCompilationContext(ImmutableMap<Artifact, BitcodeInfo> ltoBitcodeFiles) {
    this.ltoBitcodeFiles = Preconditions.checkNotNull(ltoBitcodeFiles);
  }

  /**
   * Class to hold information for a bitcode file produced by the compile action needed by the LTO
   * indexing and backend actions.
   */
  private static final class BitcodeInfo {
    private final Artifact minimizedBitcode;
    private final ImmutableList<String> copts;

    BitcodeInfo(Artifact minimizedBitcode, ImmutableList<String> copts) {
      this.minimizedBitcode = Preconditions.checkNotNull(minimizedBitcode);
      this.copts = Preconditions.checkNotNull(copts);
    }

    /** The minimized bitcode file produced by the compile and used by LTO indexing. */
    Artifact getMinimizedBitcode() {
      return minimizedBitcode;
    }

    /**
     * The compiler flags used for the compile that should also be used when finishing compilation
     * during the LTO backend.
     */
    ImmutableList<String> getCopts() {
      return copts;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      BitcodeInfo that = (BitcodeInfo) o;
      return minimizedBitcode.equals(that.minimizedBitcode) && copts.equals(that.copts);
    }

    @Override
    public int hashCode() {
      return Objects.hash(minimizedBitcode, copts);
    }
  }

  /** Builder for LtoCompilationContext. */
  public static class Builder {
    private final ImmutableMap.Builder<Artifact, BitcodeInfo> ltoBitcodeFiles =
        ImmutableMap.builder();

    public Builder() {}

    public LtoCompilationContext build() {
      ImmutableMap<Artifact, BitcodeInfo> map = ltoBitcodeFiles.build();
      if (map.isEmpty()) {
        return LtoCompilationContext.EMPTY;
      }
      return new LtoCompilationContext(map);
    }

    /** Adds a bitcode file with the corresponding minimized bitcode file and compiler flags. */
    void addBitcodeFile(
        Artifact fullBitcode, Artifact minimizedBitcode, ImmutableList<String> copts) {
      ltoBitcodeFiles.put(fullBitcode, new BitcodeInfo(minimizedBitcode, copts));
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
  Artifact getMinimizedBitcodeOrSelf(Artifact fullBitcode) {
    if (!containsBitcodeFile(fullBitcode)) {
      return fullBitcode;
    }
    return ltoBitcodeFiles.get(fullBitcode).getMinimizedBitcode();
  }

  /**
   * Gets the compiler flags corresponding to the full bitcode file, or returns an empty list if it
   * doesn't exist.
   */
  public ImmutableList<String> getCopts(Artifact fullBitcode) {
    if (!containsBitcodeFile(fullBitcode)) {
      return ImmutableList.of();
    }
    return ltoBitcodeFiles.get(fullBitcode).getCopts();
  }

  /** Whether the map of bitcode files is empty. */
  public boolean isEmpty() {
    return ltoBitcodeFiles.isEmpty();
  }

  /** The set of bitcode files recorded in the map. */
  Set<Artifact> getBitcodeFiles() {
    return ltoBitcodeFiles.keySet();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    return (o instanceof LtoCompilationContext)
        && ltoBitcodeFiles.equals(((LtoCompilationContext) o).ltoBitcodeFiles);
  }

  @Override
  public int hashCode() {
    return ltoBitcodeFiles.hashCode();
  }
}
