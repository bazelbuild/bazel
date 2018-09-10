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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineItem.ParametrizedMapFn;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CompileTimeConstant;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Factory for functions to convert a {@code T} to a commandline argument. Uses a certain convention
 * for commandline arguments (e.g., separators, and ordering of container elements).
 *
 * <p>Should only need to be created statically, and in limited quantity.
 */
public class AndroidDataConverter<T> extends ParametrizedMapFn<T> {

  /**
   * Converts parsed Android resources to the "SerializedAndroidData" format used by the Android
   * data processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ParsedAndroidResources> PARSED_RESOURCE_CONVERTER =
      AndroidDataConverter.<ParsedAndroidResources>builder(JoinerType.SEMICOLON_AMPERSAND)
          .withRoots(ParsedAndroidResources::getResourceRoots)
          .withEmpty()
          .withLabel(ParsedAndroidResources::getLabel)
          .maybeWithArtifact(ParsedAndroidResources::getSymbols)
          .build();

  /**
   * Converts compiled Android resources to the "SerializedAndroidData" format used by the Android
   * data processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ParsedAndroidResources> COMPILED_RESOURCE_CONVERTER =
      AndroidDataConverter.<ParsedAndroidResources>builder(JoinerType.SEMICOLON_AMPERSAND)
          .withRoots(ParsedAndroidResources::getResourceRoots)
          .withEmpty()
          .withLabel(ParsedAndroidResources::getLabel)
          .maybeWithArtifact(ParsedAndroidResources::getCompiledSymbols)
          .build();

  /**
   * Converts processed Android resources produced by aapt to the "DependencyAndroidData" format
   * used by the Android data processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ValidatedAndroidResources>
      AAPT_RESOURCES_AND_MANIFEST_CONVERTER =
          AndroidDataConverter.<ValidatedAndroidResources>builder(JoinerType.COLON_COMMA)
              .withRoots(ValidatedAndroidResources::getResourceRoots)
              .withEmpty()
              .withArtifact(ValidatedAndroidResources::getManifest)
              .maybeWithArtifact(ValidatedAndroidResources::getRTxt)
              .maybeWithArtifact(ValidatedAndroidResources::getSymbols)
              .build();

  /**
   * Converts processed Android resources produced by aapt2 to the "DependencyAndroidData" format
   * used by the Android data processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ValidatedAndroidResources>
      AAPT2_RESOURCES_AND_MANIFEST_CONVERTER =
          AndroidDataConverter.<ValidatedAndroidResources>builder(JoinerType.COLON_COMMA)
              .withRoots(ValidatedAndroidResources::getResourceRoots)
              .withEmpty()
              .withArtifact(ValidatedAndroidResources::getManifest)
              .maybeWithArtifact(ValidatedAndroidResources::getAapt2RTxt)
              .maybeWithArtifact(ValidatedAndroidResources::getCompiledSymbols)
              .maybeWithArtifact(ValidatedAndroidResources::getSymbols)
              .build();

  /**
   * Converts parsed Android assets to the "SerializedAndroidData" format used by the Android data
   * processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ParsedAndroidAssets> PARSED_ASSET_CONVERTER =
      AndroidDataConverter.<ParsedAndroidAssets>builder(JoinerType.SEMICOLON_AMPERSAND)
          .withEmpty()
          .withRoots(ParsedAndroidAssets::getAssetRoots)
          .withLabel(ParsedAndroidAssets::getLabel)
          .maybeWithArtifact(ParsedAndroidAssets::getSymbols)
          .build();

  /**
   * Converts compiled Android assets to the "SerializedAndroidData" format used by the Android data
   * processing actions.
   */
  @AutoCodec
  static final AndroidDataConverter<ParsedAndroidAssets> COMPILED_ASSET_CONVERTER =
      AndroidDataConverter.<ParsedAndroidAssets>builder(JoinerType.SEMICOLON_AMPERSAND)
          .withEmpty()
          .withRoots(ParsedAndroidAssets::getAssetRoots)
          .withLabel(ParsedAndroidAssets::getLabel)
          .maybeWithArtifact(ParsedAndroidAssets::getCompiledSymbols)
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

  // We must override equals and hashCode as per the contract of ParametrizedMapFn, but we
  // statically create a very small number of these objects, so we know that reference equality is
  // enough.
  @Override
  public boolean equals(Object obj) {
    return this == obj;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
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
   *
   * <p>Additionally, the resulting {@link AndroidDataConverter} object should be annotated with
   * {@link AutoCodec} (and, if relevant, {@link
   * com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization}.
   */
  public static <T> Builder<T> builder(JoinerType joinerType) {
    return new Builder<>(joinerType);
  }

  public VectorArg<String> getVectorArg(NestedSet<? extends T> values) {
    return VectorArg.join(joinerType.listSeparator).each(values).mapped(this);
  }

  public VectorArg<String> getVectorArgForEach(
      @CompileTimeConstant String arg, NestedSet<? extends T> values) {
    return VectorArg.addBefore(arg).each(values).mapped(this);
  }

  static class Builder<T> {
    private final ImmutableList.Builder<Function<T, String>> inner = ImmutableList.builder();
    private final JoinerType joinerType;

    private Builder(JoinerType joinerType) {
      this.joinerType = joinerType;
    }

    Builder<T> withRoots(Function<T, ImmutableList<PathFragment>> rootsFunction) {
      return with(t -> rootsToString(rootsFunction.apply(t)));
    }

    Builder<T> withArtifact(Function<T, Artifact> artifactFunction) {
      return with(t -> artifactFunction.apply(t).getExecPathString());
    }

    Builder<T> withEmpty() {
      return with(t -> "");
    }

    Builder<T> maybeWithArtifact(Function<T, Artifact> nullableArtifactFunction) {
      return with(
          t -> {
            @Nullable Artifact artifact = nullableArtifactFunction.apply(t);
            return artifact == null ? "" : artifact.getExecPathString();
          });
    }

    Builder<T> withLabel(Function<T, Label> labelFunction) {
      // Escape labels, since they are known to contain separating characters (specifically, ':').
      // Anonymous inner class for serialization.
      return with(t -> joinerType.escape(labelFunction.apply(t).toString()));
    }

    Builder<T> with(Function<T, String> stringFunction) {
      inner.add(stringFunction);
      return this;
    }

    AndroidDataConverter<T> build() {
      return new AndroidDataConverter<>(inner.build(), joinerType);
    }
  }

  @VisibleForTesting
  public static String rootsToString(ImmutableList<PathFragment> roots) {
    return roots.stream().map(PathFragment::toString).collect(Collectors.joining("#"));
  }
}
