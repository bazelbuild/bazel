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
package com.google.devtools.build.lib.rules.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.ToArg.Includes;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Factory for functions to convert a {@link ResourceContainer} to a commandline argument, or a
 * collection of artifacts. Uses a certain convention for commandline arguments (e.g., separators,
 * and ordering of container elements).
 *
 * @deprecated Use {@link AndroidDataConverter} instead.
 */
@VisibleForTesting
@Deprecated
public class ResourceContainerConverter {

  static Builder builder() {
    return new Builder();
  }

  static class ToArg extends CommandLineItem.ParametrizedMapFn<ValidatedAndroidData> {

    private final Set<Includes> includes;
    private final SeparatorType separatorType;
    private final Joiner argJoiner;
    private final Function<String, String> escaper;

    enum Includes {
      ResourceRoots,
      Label,
      Manifest,
      RTxt,
      SymbolsBin,
      CompiledSymbols,
      StaticLibrary,
      Aapt2RTxt
    }

    enum SeparatorType {
      COLON_COMMA,
      SEMICOLON_AMPERSAND
    }

    ToArg(Builder builder) {
      this.includes = Sets.immutableEnumSet(builder.includes);
      this.separatorType = builder.separatorType;

      switch (separatorType) {
        case COLON_COMMA:
          argJoiner = Joiner.on(":");
          // We currently use ":" to separate components of an argument and "," to separate
          // arguments in a list of arguments. Those characters require escaping if used in a label
          // (part of the set of allowed characters in a label).
          if (includes.contains(Includes.Label)) {
            escaper = (String input) -> input.replace(":", "\\:").replace(",", "\\,");
          } else {
            escaper = Functions.identity();
          }
          break;
        case SEMICOLON_AMPERSAND:
          argJoiner = Joiner.on(";");
          escaper = Functions.identity();
          break;
        default:
          throw new IllegalStateException("Unknown separator type " + separatorType);
      }
    }

    @Override
    public void expandToCommandLine(ValidatedAndroidData container, Consumer<String> args) {
      args.accept(map(container));
    }

    String map(ValidatedAndroidData container) {
      ImmutableList.Builder<String> cmdPieces = ImmutableList.builder();
      if (includes.contains(Includes.ResourceRoots)) {
        cmdPieces.add(convertRoots(container.getResourceRoots()));
        cmdPieces.add(convertRoots(container.getAssetRoots()));
      }
      if (includes.contains(Includes.Label)) {
        cmdPieces.add(escaper.apply(container.getLabel().toString()));
      }
      if (includes.contains(Includes.Manifest)) {
        cmdPieces.add(container.getManifest().getExecPathString());
      }
      if (includes.contains(Includes.RTxt)) {
        cmdPieces.add(container.getRTxt() == null ? "" : container.getRTxt().getExecPathString());
      }
      if (includes.contains(Includes.Aapt2RTxt)) {
        cmdPieces.add(
            container.getAapt2RTxt() == null ? "" : container.getAapt2RTxt().getExecPathString());
      }
      if (includes.contains(Includes.StaticLibrary)) {
        cmdPieces.add(
            container.getStaticLibrary() == null
                ? ""
                : container.getStaticLibrary().getExecPathString());
      }
      if (includes.contains(Includes.CompiledSymbols)) {
        cmdPieces.add(
            container.getCompiledSymbols() == null
                ? ""
                : container.getCompiledSymbols().getExecPathString());
      }
      if (includes.contains(Includes.SymbolsBin)) {
        cmdPieces.add(
            container.getSymbols() == null ? "" : container.getSymbols().getExecPathString());
      }
      return argJoiner.join(cmdPieces.build());
    }

    String listSeparator() {
      switch (separatorType) {
        case COLON_COMMA:
          return ",";
        case SEMICOLON_AMPERSAND:
          return "&";
        default:
          Preconditions.checkState(false, "Unknown separator type " + separatorType);
          return null;
      }
    }

    @Override
    public int maxInstancesAllowed() {
      // This is the max number of resource converters we expect to statically
      // construct for any given blaze instance.
      // Do not increase recklessly.
      return 10;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      ToArg toArg = (ToArg) o;
      return includes.equals(toArg.includes) && separatorType == toArg.separatorType;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(includes, separatorType);
    }
  }

  static class Builder {

    private final Set<Includes> includes = new HashSet<>();
    private ToArg.SeparatorType separatorType;

    Builder() {}

    Builder include(Includes include) {
      includes.add(include);
      return this;
    }

    Builder withSeparator(ToArg.SeparatorType type) {
      separatorType = type;
      return this;
    }

    ToArg toArgConverter() {
      return new ToArg(this);
    }
  }

  @VisibleForTesting
  public static String convertRoots(Iterable<PathFragment> roots) {
    return Joiner.on("#").join(Iterators.transform(roots.iterator(), Functions.toStringFunction()));
  }

  /**
   * Convert ResourceDependencies to commandline args and artifacts, assuming the commandline
   * arguments should be split into direct deps and transitive deps.
   */
  static void addToCommandLine(
      ResourceDependencies dependencies, CustomCommandLine.Builder cmdBuilder, ToArg toArg) {
    cmdBuilder.addAll(
        "--data",
        VectorArg.join(toArg.listSeparator())
            .each(dependencies.getTransitiveResourceContainers())
            .mapped(toArg));
    cmdBuilder.addAll(
        "--directData",
        VectorArg.join(toArg.listSeparator())
            .each(dependencies.getDirectResourceContainers())
            .mapped(toArg));
  }
}
