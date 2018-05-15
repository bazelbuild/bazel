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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineItem.ParametrizedMapFn;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Factory for functions to convert a {@code T} to a commandline argument. Uses a certain convention
 * for commandline arguments (e.g., separators, and ordering of container elements).
 */
public class AndroidDataConverter<T> extends ParametrizedMapFn<T> {

  /**
   * Converts Android data to the "SerializedAndroidData" format used by the Android data processing
   * actions.
   */
  static final AndroidDataConverter<MergableAndroidData> MERGABLE_DATA_CONVERTER =
      AndroidDataConverter.<MergableAndroidData>builder(JoinerType.SEMICOLON_AMPERSAND)
          .withRoots(MergableAndroidData::getResourceRoots)
          .withRoots(MergableAndroidData::getAssetRoots)
          .withLabel(MergableAndroidData::getLabel)
          .withArtifact(MergableAndroidData::getSymbols)
          .build();

  /** Indicates the type of joiner between options expected by the command line. */
  public enum JoinerType {
    COLON_COMMA(":", ","),
    SEMICOLON_AMPERSAND(";", "&");

    private final String itemSeparator;
    private final String listSeparator;

    JoinerType(String itemSeparator, String listSeparator) {
      this.itemSeparator = itemSeparator;
      this.listSeparator = listSeparator;
    }

    private String escape(String string) {
      return string
          .replace(itemSeparator, "\\" + itemSeparator)
          .replace(listSeparator, "\\" + listSeparator);
    }
  }

  private final ImmutableList<Function<T, String>> suppliers;
  private final JoinerType joinerType;

  private AndroidDataConverter(
      ImmutableList<Function<T, String>> suppliers, JoinerType joinerType) {
    this.suppliers = suppliers;
    this.joinerType = joinerType;
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof AndroidDataConverter)) {
      return false;
    }

    AndroidDataConverter<?> other = (AndroidDataConverter) obj;
    return suppliers.equals(other.suppliers) && joinerType.equals(other.joinerType);
  }

  @Override
  public int hashCode() {
    return Objects.hash(suppliers, joinerType);
  }

  @Override
  public int maxInstancesAllowed() {
    // This is the max number of resource converters we expect to statically
    // construct for any given blaze instance.
    // Do not increase recklessly.
    return 10;
  }

  @Override
  public void expandToCommandLine(T t, Consumer<String> args) {
    args.accept(map(t));
  }

  public String map(T t) {
    return suppliers
        .stream()
        .map(s -> (s.apply(t)))
        .collect(Collectors.joining(joinerType.itemSeparator));
  }

  /**
   * Creates a builder for a new {@link AndroidDataConverter}.
   *
   * <p>Because of how Bazel handles these objects, call this method *only* as part of creating a
   * static final field.
   */
  public static <T> Builder<T> builder(JoinerType joinerType) {
    return new Builder<>(joinerType);
  }

  public void addDepsToCommandLine(
      CustomCommandLine.Builder cmdBuilder,
      NestedSet<? extends T> direct,
      NestedSet<? extends T> transitive) {
    cmdBuilder.addAll("--data", getVectorArg(transitive));
    cmdBuilder.addAll("--directData", getVectorArg(direct));
  }

  public VectorArg<String> getVectorArg(NestedSet<? extends T> values) {
    return VectorArg.join(joinerType.listSeparator).each(values).mapped(this);
  }

  static class Builder<T> {
    private final ImmutableList.Builder<Function<T, String>> inner = ImmutableList.builder();
    private final JoinerType joinerType;

    private Builder(JoinerType joinerType) {
      this.joinerType = joinerType;
    }

    Builder<T> withRoots(Function<T, ImmutableList<PathFragment>> rootsFunction) {
      return with(new Function<T, String>() {
        @Override
        public String apply(T t) {
          return rootsToString(rootsFunction.apply(t));
        }
      });
    }

    Builder<T> withArtifact(Function<T, Artifact> artifactFunction) {
      return with(new Function<T, String>() {
        @Override
        public String apply(T t) {
          return artifactFunction.apply(t).getExecPathString();
        }
      });
    }

    Builder<T> withLabel(Function<T, Label> labelFunction) {
      // Escape labels, since they are known to contain separating characters (specifically, ':').
      return with(new Function<T, String>() {
        @Override
        public String apply(T t) {
          return joinerType.escape(labelFunction.apply(t).toString());
        }
      });
    }

    Builder<T> with(Function<T, String> stringFunction) {
      inner.add(stringFunction);
      return this;
    }

    AndroidDataConverter<T> build() {
      return new AndroidDataConverter<>(inner.build(), joinerType);
    }
  }

  static String rootsToString(ImmutableList<PathFragment> roots) {
    return roots.stream().map(PathFragment::toString).collect(Collectors.joining("#"));
  }
}
