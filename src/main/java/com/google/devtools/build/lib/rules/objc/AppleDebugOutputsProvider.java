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
package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 * A provider that holds debug outputs of an Apple binary rule.
 *
 * <p>This provider has no native interface and is intended to be read in Skylark code.
 *
 * <p>The only field it has is {@code output_map}, which is a dictionary of: { arch: { output_type:
 * Artifact, output_type: Artifact, ... } }
 *
 * <p>Where {@code arch} is any Apple architecture such as "arm64" or "armv7", {@code output_type}
 * can currently be "bitcode_symbols" or "dsym_binary", and the artifact is an instance of the
 * {@link Artifact} class.
 *
 * <p>Example: { "arm64": { "bitcode_symbols": Artifact, "dsym_binary": Artifact } }
 */
@Immutable
public final class AppleDebugOutputsProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {

  /** Expected types of debug outputs. */
  enum OutputType {

    /** A Bitcode symbol map, per architecture. */
    BITCODE_SYMBOLS,

    /** A single-architecture DWARF binary with debug symbols. */
    DSYM_BINARY;

    @Override
    public String toString() {
      return name().toLowerCase();
    }
  }

  public static final ClassObjectConstructor SKYLARK_PROVIDER =
      new NativeClassObjectConstructor("AppleDebugOutputs") { };

  /**
   * Creates a new provider instance.
   *
   * @param map a map of
   *     <pre>{@code
   * {
   *   arch: { output_type: Artifact, output_type: Artifact, ... },
   * }
   * }</pre>
   *     Where:
   *     <ul>
   *       <li>arch - {@link String}, any Apple supported architecture (e.g. arm64, x86_64)
   *       <li>output_type - an instance of {@link OutputType}
   *     </ul>
   */
  private AppleDebugOutputsProvider(ImmutableMap<String, ImmutableMap<String, Artifact>> map) {
    super(SKYLARK_PROVIDER, ImmutableMap.<String, Object>of("outputs_map", map));
  }

  /** A builder for {@link AppleDebugOutputsProvider}. */
  public static class Builder {
    private final HashMap<String, HashMap<String, Artifact>> outputsByArch = Maps.newHashMap();

    private Builder() {}

    public static Builder create() {
      return new Builder();
    }

    /**
     * Adds an output to the provider.
     *
     * @param arch any Apple architecture string, e.g. arm64, armv7.
     * @param outputType {@link OutputType} corresponding to the artifact.
     * @param artifact an {@link Artifact} that contains debug information.
     * @return this builder.
     */
    public Builder addOutput(String arch, OutputType outputType, Artifact artifact) {
      if (!outputsByArch.containsKey(arch)) {
        outputsByArch.put(arch, new HashMap<String, Artifact>());
      }

      outputsByArch.get(arch).put(outputType.toString(), artifact);
      return this;
    }

    public AppleDebugOutputsProvider build() {
      ImmutableMap.Builder<String, ImmutableMap<String, Artifact>> builder = ImmutableMap.builder();

      for (Entry<String, HashMap<String, Artifact>> e : outputsByArch.entrySet()) {
        builder.put(e.getKey(), ImmutableMap.copyOf(e.getValue()));
      }

      return new AppleDebugOutputsProvider(builder.build());
    }
  }
}
