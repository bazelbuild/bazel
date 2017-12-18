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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;

/**
 * Factory for functions to convert a {@link ResourceContainer} to a commandline argument, or a
 * collection of artifacts. Uses a certain convention for commandline arguments (e.g., separators,
 * and ordering of container elements).
 */
@VisibleForTesting
public class ResourceContainerConverter {

  static Builder builder() {
    return new Builder();
  }

  interface ToArg extends Function<ResourceContainer, String> {

    String listSeparator();
  }

  static class Builder {

    private boolean includeResourceRoots;
    private boolean includeLabel;
    private boolean includeManifest;
    private boolean includeRTxt;
    private boolean includeSymbolsBin;
    private boolean includeCompiledSymbols;
    private boolean includeStaticLibrary;
    private boolean includeAapt2RTxt;
    private SeparatorType separatorType;
    private Joiner argJoiner;
    private Function<String, String> escaper = Functions.identity();

    enum SeparatorType {
      COLON_COMMA,
      SEMICOLON_AMPERSAND
    }

    Builder() {}

    Builder includeAapt2RTxt() {
      includeAapt2RTxt = true;
      return this;
    }

    Builder includeStaticLibrary() {
      includeStaticLibrary = true;
      return this;
    }

    Builder includeResourceRoots() {
      includeResourceRoots = true;
      return this;
    }

    Builder includeLabel() {
      includeLabel = true;
      return this;
    }

    Builder includeManifest() {
      includeManifest = true;
      return this;
    }

    Builder includeRTxt() {
      includeRTxt = true;
      return this;
    }

    Builder includeSymbolsBin() {
      includeSymbolsBin = true;
      return this;
    }

    Builder includeCompiledSymbols() {
      includeCompiledSymbols = true;
      return this;
    }

    Builder withSeparator(SeparatorType type) {
      separatorType = type;
      return this;
    }

    ToArg toArgConverter() {
      switch (separatorType) {
        case COLON_COMMA:
          argJoiner = Joiner.on(":");
          // We currently use ":" to separate components of an argument and "," to separate
          // arguments in a list of arguments. Those characters require escaping if used in a label
          // (part of the set of allowed characters in a label).
          if (includeLabel) {
            escaper = (String input) -> input.replace(":", "\\:").replace(",", "\\,");
          }
          break;
        case SEMICOLON_AMPERSAND:
          argJoiner = Joiner.on(";");
          break;
        default:
          Preconditions.checkState(false, "Unknown separator type " + separatorType);
          break;
      }

      return new ToArg() {
        @Override
        public String apply(ResourceContainer container) {
          ImmutableList.Builder<String> cmdPieces = ImmutableList.builder();
          if (includeResourceRoots) {
            cmdPieces.add(convertRoots(container, ResourceType.RESOURCES));
            cmdPieces.add(convertRoots(container, ResourceType.ASSETS));
          }
          if (includeLabel) {
            cmdPieces.add(escaper.apply(container.getLabel().toString()));
          }
          if (includeManifest) {
            cmdPieces.add(container.getManifest().getExecPathString());
          }
          if (includeRTxt) {
            cmdPieces.add(
                container.getRTxt() == null ? "" : container.getRTxt().getExecPathString());
          }
          if (includeAapt2RTxt) {
            cmdPieces.add(
                container.getAapt2RTxt() == null
                    ? ""
                    : container.getAapt2RTxt().getExecPathString());
          }
          if (includeStaticLibrary) {
            cmdPieces.add(
                container.getStaticLibrary() == null
                    ? ""
                    : container.getStaticLibrary().getExecPathString());
          }
          if (includeCompiledSymbols) {
            cmdPieces.add(
                container.getCompiledSymbols() == null
                    ? ""
                    : container.getCompiledSymbols().getExecPathString());
          }
          if (includeSymbolsBin) {
            cmdPieces.add(
                container.getSymbols() == null ? "" : container.getSymbols().getExecPathString());
          }
          return argJoiner.join(cmdPieces.build());
        }

        @Override
        public String listSeparator() {
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
      };
    }
  }

  @VisibleForTesting
  public static String convertRoots(ResourceContainer container, ResourceType resourceType) {
    return Joiner.on("#")
        .join(
            Iterators.transform(
                container.getRoots(resourceType).iterator(), Functions.toStringFunction()));
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
