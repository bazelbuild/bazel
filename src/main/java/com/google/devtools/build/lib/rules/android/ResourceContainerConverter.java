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
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import javax.annotation.Nullable;

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

  interface ToArtifacts extends Function<ResourceContainer, NestedSet<Artifact>> {

  }

  static class Builder {

    private boolean includeResourceRoots;
    private boolean includeLabel;
    private boolean includeManifest;
    private boolean includeRTxt;
    private boolean includeSymbolsBin;
    private boolean includeStaticLibrary;
    private boolean includeAapt2RTxt;
    private SeparatorType separatorType;
    private Joiner argJoiner;
    private Function<String, String> escaper = Functions.identity();


    enum SeparatorType {
      COLON_COMMA,
      SEMICOLON_AMPERSAND
    }

    Builder() {
    }

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

    ToArtifacts toArtifactConverter() {
      return new ToArtifacts() {
        @Override
        public NestedSet<Artifact> apply(ResourceContainer container) {
          NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.naiveLinkOrder();
          if (includeResourceRoots) {
            artifacts.addAll(container.getArtifacts());
          }
          if (includeManifest) {
            addIfNotNull(container.getManifest(), artifacts);
          }
          if (includeRTxt) {
            addIfNotNull(container.getRTxt(), artifacts);
          }
          if (includeSymbolsBin) {
            addIfNotNull(container.getSymbols(), artifacts);
          }
          if (includeAapt2RTxt) {
            addIfNotNull(container.getAapt2RTxt(), artifacts);
          }
          if (includeStaticLibrary) {
            addIfNotNull(container.getStaticLibrary(), artifacts);
          }
          return artifacts.build();
        }
      };
    }
  }

  private static void addIfNotNull(
      @Nullable Artifact artifact, NestedSetBuilder<Artifact> artifacts) {
    if (artifact != null) {
      artifacts.add(artifact);
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
  static void convertDependencies(
      ResourceDependencies dependencies,
      CustomCommandLine.Builder cmdBuilder,
      NestedSetBuilder<Artifact> inputs,
      ToArg toArg,
      ToArtifacts toArtifacts) {

    if (dependencies != null) {
      if (!dependencies.getTransitiveResources().isEmpty()) {
        cmdBuilder.add(
            "--data",
            VectorArg.of(dependencies.getTransitiveResources())
                .joinWith(toArg.listSeparator())
                .mapEach(toArg));
      }
      if (!dependencies.getDirectResources().isEmpty()) {
        cmdBuilder.add(
            "--directData",
            VectorArg.of(dependencies.getDirectResources())
                .joinWith(toArg.listSeparator())
                .mapEach(toArg));
      }
      // This flattens the nested set. Since each ResourceContainer needs to be transformed into
      // Artifacts, and the NestedSetBuilder.wrap doesn't support lazy Iterator evaluation
      // and SpawnActionBuilder.addInputs evaluates Iterables, it becomes necessary to make the
      // best effort and let it get flattened.
      inputs.addTransitive(
          NestedSetBuilder.wrap(
              Order.NAIVE_LINK_ORDER,
              FluentIterable.from(dependencies.getResources()).transformAndConcat(toArtifacts)));
    }
  }
}
