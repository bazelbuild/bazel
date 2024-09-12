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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.MissingExpansionException;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehaviorWithoutError;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.SingleStringArgFormatter;
import com.google.devtools.build.lib.analysis.actions.PathMappers;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.DirectoryExpander;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IllegalFormatException;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** Supports ctx.actions.args() from Starlark. */
public class StarlarkCustomCommandLine extends CommandLine {

  private static final Joiner LINE_JOINER = Joiner.on("\n").skipNulls();
  private static final Joiner FIELD_JOINER = Joiner.on(": ").skipNulls();

  // Used to distinguish command line arguments that are potentially subject to special default
  // stringification (such as Artifacts when path mapped or Labels when not main repo labels) from
  // strings that happen to be identical to their string representations.
  enum StringificationType {
    DEFAULT,
    FILE,
    LABEL
  }

  @AutoCodec
  static final class VectorArg {
    private static final Interner<VectorArg> interner = BlazeInterners.newStrongInterner();

    private static final int HAS_LOCATION = 1;
    // Deleted HAS_MAP_ALL = 1 << 1;
    private static final int HAS_MAP_EACH = 1 << 2;
    private static final int IS_NESTED_SET = 1 << 3;
    private static final int EXPAND_DIRECTORIES = 1 << 4;
    private static final int UNIQUIFY = 1 << 5;
    private static final int OMIT_IF_EMPTY = 1 << 6;
    private static final int HAS_ARG_NAME = 1 << 7;
    private static final int HAS_FORMAT_EACH = 1 << 8;
    private static final int HAS_BEFORE_EACH = 1 << 9;
    private static final int HAS_JOIN_WITH = 1 << 10;
    private static final int HAS_FORMAT_JOINED = 1 << 11;
    private static final int HAS_TERMINATE_WITH = 1 << 12;

    private static final UUID EXPAND_DIRECTORIES_UUID =
        UUID.fromString("9d7520d2-a187-11e8-98d0-529269fb1459");
    private static final UUID UNIQUIFY_UUID =
        UUID.fromString("7f494c3e-faea-4498-a521-5d3bc6ee19eb");
    private static final UUID OMIT_IF_EMPTY_UUID =
        UUID.fromString("923206f1-6474-4a8f-b30f-4dd3143622e6");
    private static final UUID ARG_NAME_UUID =
        UUID.fromString("2bc00382-7199-46ec-ad52-1556577cde1a");
    private static final UUID FORMAT_EACH_UUID =
        UUID.fromString("8e974aec-df07-4a51-9418-f4c1172b4045");
    private static final UUID BEFORE_EACH_UUID =
        UUID.fromString("f7e101bc-644d-4277-8562-6515ad55a988");
    private static final UUID JOIN_WITH_UUID =
        UUID.fromString("c227dbd3-edad-454e-bc8a-c9b5ba1c38a3");
    private static final UUID FORMAT_JOINED_UUID =
        UUID.fromString("528af376-4233-4c27-be4d-b0ff24ed68db");
    private static final UUID TERMINATE_WITH_UUID =
        UUID.fromString("a4e5e090-0dbd-4d41-899a-77cfbba58655");

    private final int features;
    private final StringificationType stringificationType;

    private VectorArg(int features, StringificationType stringificationType) {
      this.features = features;
      this.stringificationType = stringificationType;
    }

    @VisibleForSerialization
    @AutoCodec.Instantiator
    static VectorArg create(int features, StringificationType stringificationType) {
      return interner.intern(new VectorArg(features, stringificationType));
    }

    private static void push(
        List<Object> arguments, Builder arg, StarlarkSemantics starlarkSemantics) {
      int features = 0;
      features |= arg.mapEach != null ? HAS_MAP_EACH : 0;
      features |= arg.nestedSet != null ? IS_NESTED_SET : 0;
      features |= arg.expandDirectories ? EXPAND_DIRECTORIES : 0;
      features |= arg.uniquify ? UNIQUIFY : 0;
      features |= arg.omitIfEmpty ? OMIT_IF_EMPTY : 0;
      features |= arg.argName != null ? HAS_ARG_NAME : 0;
      features |= arg.formatEach != null ? HAS_FORMAT_EACH : 0;
      features |= arg.beforeEach != null ? HAS_BEFORE_EACH : 0;
      features |= arg.joinWith != null ? HAS_JOIN_WITH : 0;
      features |= arg.formatJoined != null ? HAS_FORMAT_JOINED : 0;
      features |= arg.terminateWith != null ? HAS_TERMINATE_WITH : 0;
      boolean hasLocation =
          arg.location != null
              && (features & (HAS_FORMAT_EACH | HAS_FORMAT_JOINED | HAS_MAP_EACH)) != 0;
      features |= hasLocation ? HAS_LOCATION : 0;
      VectorArg vectorArg = VectorArg.create(features, arg.nestedSetStringificationType);
      arguments.add(vectorArg);
      if (hasLocation) {
        arguments.add(arg.location);
      }
      if (arg.mapEach != null) {
        arguments.add(arg.mapEach);
        arguments.add(starlarkSemantics);
      }
      if (arg.nestedSet != null) {
        arguments.add(arg.nestedSet);
      } else {
        List<?> list = arg.list;
        int count = list.size();
        arguments.add(count);
        for (int i = 0; i < count; ++i) {
          arguments.add(list.get(i));
        }
      }
      if (arg.argName != null) {
        arguments.add(arg.argName);
      }
      if (arg.formatEach != null) {
        arguments.add(arg.formatEach);
      }
      if (arg.beforeEach != null) {
        arguments.add(arg.beforeEach);
      }
      if (arg.joinWith != null) {
        arguments.add(arg.joinWith);
      }
      if (arg.formatJoined != null) {
        arguments.add(arg.formatJoined);
      }
      if (arg.terminateWith != null) {
        arguments.add(arg.terminateWith);
      }
    }

    private int eval(
        List<Object> arguments,
        int argi,
        List<String> builder,
        @Nullable ArtifactExpander artifactExpander,
        PathMapper pathMapper,
        @Nullable RepositoryMapping mainRepoMapping)
        throws CommandLineExpansionException, InterruptedException {
      final Location location =
          ((features & HAS_LOCATION) != 0) ? (Location) arguments.get(argi++) : null;
      final List<Object> originalValues;
      StarlarkCallable mapEach =
          ((features & HAS_MAP_EACH) != 0) ? (StarlarkCallable) arguments.get(argi++) : null;
      StarlarkSemantics starlarkSemantics =
          ((features & HAS_MAP_EACH) != 0) ? (StarlarkSemantics) arguments.get(argi++) : null;
      if ((features & IS_NESTED_SET) != 0) {
        @SuppressWarnings("unchecked")
        NestedSet<Object> nestedSet = (NestedSet<Object>) arguments.get(argi++);
        originalValues = nestedSet.toList();
      } else {
        int count = (Integer) arguments.get(argi++);
        originalValues = arguments.subList(argi, argi + count);
        argi += count;
      }
      List<Object> expandedValues =
          maybeExpandDirectories(artifactExpander, originalValues, pathMapper);
      List<String> stringValues;
      if (mapEach != null) {
        stringValues = new ArrayList<>(expandedValues.size());
        applyMapEach(
            mapEach,
            expandedValues,
            stringValues::add,
            location,
            artifactExpander,
            pathMapper,
            starlarkSemantics);
      } else {
        int count = expandedValues.size();
        stringValues = new ArrayList<>(expandedValues.size());
        for (int i = 0; i < count; ++i) {
          stringValues.add(expandToCommandLine(expandedValues.get(i), pathMapper, mainRepoMapping));
        }
      }
      // It's safe to uniquify at this stage, any transformations after this
      // will ensure continued uniqueness of the values
      if ((features & UNIQUIFY) != 0) {
        int count = stringValues.size();
        HashSet<String> seen = Sets.newHashSetWithExpectedSize(count);
        int addIndex = 0;
        for (int i = 0; i < count; ++i) {
          String val = stringValues.get(i);
          if (seen.add(val)) {
            stringValues.set(addIndex++, val);
          }
        }
        stringValues = stringValues.subList(0, addIndex);
      }
      boolean isEmptyAndShouldOmit = stringValues.isEmpty() && (features & OMIT_IF_EMPTY) != 0;
      if ((features & HAS_ARG_NAME) != 0) {
        String argName = (String) arguments.get(argi++);
        if (!isEmptyAndShouldOmit) {
          builder.add(argName);
        }
      }
      if ((features & HAS_FORMAT_EACH) != 0) {
        String formatStr = (String) arguments.get(argi++);
        try {
          int count = stringValues.size();
          for (int i = 0; i < count; ++i) {
            stringValues.set(i, SingleStringArgFormatter.format(formatStr, stringValues.get(i)));
          }
        } catch (IllegalFormatException e) {
          throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, null));
        }
      }
      if ((features & HAS_BEFORE_EACH) != 0) {
        String beforeEach = (String) arguments.get(argi++);
        int count = stringValues.size();
        for (int i = 0; i < count; ++i) {
          builder.add(beforeEach);
          builder.add(stringValues.get(i));
        }
      } else if ((features & HAS_JOIN_WITH) != 0) {
        String joinWith = (String) arguments.get(argi++);
        String formatJoined =
            ((features & HAS_FORMAT_JOINED) != 0) ? (String) arguments.get(argi++) : null;
        if (!isEmptyAndShouldOmit) {
          String result = Joiner.on(joinWith).join(stringValues);
          if (formatJoined != null) {
            try {
              result = SingleStringArgFormatter.format(formatJoined, result);
            } catch (IllegalFormatException e) {
              throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, null));
            }
          }
          builder.add(result);
        }
      } else {
        builder.addAll(stringValues);
      }
      if ((features & HAS_TERMINATE_WITH) != 0) {
        String terminateWith = (String) arguments.get(argi++);
        if (!isEmptyAndShouldOmit) {
          builder.add(terminateWith);
        }
      }
      return argi;
    }

    /**
     * Expands the directories if {@code expand_directories} feature is enabled and a
     * ArtifactExpander is available.
     *
     * <p>Technically, we should always expand the directories if the feature is requested, however
     * we cannot do that in the absence of the {@link ArtifactExpander}.
     */
    private List<Object> maybeExpandDirectories(
        @Nullable ArtifactExpander artifactExpander,
        List<Object> originalValues,
        PathMapper pathMapper)
        throws CommandLineExpansionException {
      if ((features & EXPAND_DIRECTORIES) == 0
          || artifactExpander == null
          || !hasDirectory(originalValues)) {
        return originalValues;
      }

      return expandDirectories(artifactExpander, originalValues, pathMapper);
    }

    private static boolean hasDirectory(List<Object> originalValues) {
      int n = originalValues.size();
      for (int i = 0; i < n; ++i) {
        Object object = originalValues.get(i);
        if (isDirectory(object)) {
          return true;
        }
      }
      return false;
    }

    private static boolean isDirectory(Object object) {
      return object instanceof Artifact && ((Artifact) object).isDirectory();
    }

    private static List<Object> expandDirectories(
        Artifact.ArtifactExpander artifactExpander,
        List<Object> originalValues,
        PathMapper pathMapper)
        throws CommandLineExpansionException {
      List<Object> expandedValues = new ArrayList<>(originalValues.size());
      for (Object object : originalValues) {
        if (isDirectory(object)) {
          Artifact artifact = (Artifact) object;
          if (artifact.isTreeArtifact()) {
            artifactExpander.expand((Artifact) object, expandedValues);
          } else if (artifact.isFileset()) {
            expandFileset(artifactExpander, artifact, expandedValues, pathMapper);
          } else {
            throw new AssertionError("Unknown artifact type.");
          }
        } else {
          expandedValues.add(object);
        }
      }
      return expandedValues;
    }

    private static void expandFileset(
        Artifact.ArtifactExpander artifactExpander,
        Artifact fileset,
        List<Object> expandedValues,
        PathMapper pathMapper)
        throws CommandLineExpansionException {
      ImmutableList<FilesetOutputSymlink> expandedFileSet;
      try {
        expandedFileSet = artifactExpander.getFileset(fileset);
      } catch (MissingExpansionException e) {
        throw new CommandLineExpansionException(
            String.format(
                "Could not expand fileset: %s. Did you forget to add it as an input of the"
                    + " action?",
                fileset),
            e);
      }
      FilesetManifest filesetManifest =
          FilesetManifest.constructFilesetManifestWithoutError(
              expandedFileSet, fileset.getExecPath(), RelativeSymlinkBehaviorWithoutError.IGNORE);
      for (PathFragment relativePath : filesetManifest.getEntries().keySet()) {
        PathFragment mappedRelativePath = pathMapper.map(relativePath);
        expandedValues.add(new FilesetSymlinkFile(fileset, mappedRelativePath));
      }
    }

    private int addToFingerprint(
        List<Object> arguments,
        int argi,
        ActionKeyContext actionKeyContext,
        Fingerprint fingerprint,
        @Nullable ArtifactExpander artifactExpander,
        CoreOptions.OutputPathsMode outputPathsMode)
        throws CommandLineExpansionException, InterruptedException {
      final Location location =
          ((features & HAS_LOCATION) != 0) ? (Location) arguments.get(argi++) : null;
      StarlarkCallable mapEach =
          ((features & HAS_MAP_EACH) != 0) ? (StarlarkCallable) arguments.get(argi++) : null;
      StarlarkSemantics starlarkSemantics =
          ((features & HAS_MAP_EACH) != 0) ? (StarlarkSemantics) arguments.get(argi++) : null;

      // NestedSets and lists never result in the same fingerprint as the
      // ActionKeyContext#addNestedSetToFingerprint call below always adds the order of the
      // NestedSet to the fingerprint.
      //
      // Path mapping may affect the default stringification of Artifact instances at execution
      // time, but the effect of path mapping on an individual command line element is a pure
      // function of:
      // * whether the element is of type Artifact (FileApi in Starlark), which is fingerprinted
      //   via the elementType UUID below;
      // * the path of the artifact, which is fingerprinted via its default string representation
      //   below;
      // * the paths and possibly the digests of all input artifacts as well as the path mapping
      //   mode, which are fingerprinted by SpawnAction.
      // It is thus safe to ignore pathMapper below for anything that relies on the default
      // stringification behavior (which excludes custom mapEach functions).
      if ((features & IS_NESTED_SET) != 0) {
        NestedSet<?> values = (NestedSet<?>) arguments.get(argi++);
        if (mapEach != null) {
          // mapEach functions do not rely on default stringification behavior, so we can omit
          // fingerprinting stringificationType here.
          CommandLineItemMapEachAdaptor commandLineItemMapFn =
              new CommandLineItemMapEachAdaptor(
                  mapEach,
                  location,
                  starlarkSemantics,
                  (features & EXPAND_DIRECTORIES) != 0 ? artifactExpander : null,
                  outputPathsMode);
          try {
            actionKeyContext.addNestedSetToFingerprint(commandLineItemMapFn, fingerprint, values);
          } finally {
            // The cache holds an entry for a NestedSet for every (map_fn, hasArtifactExpanderBit,
            // pathMapperCacheKey).
            // Clearing the artifactExpander itself saves us from storing the contents of it in the
            // cache keys (it is no longer needed after we evaluate the value).
            // NestedSet cache is cleared after every build, which means that the artifactExpander
            // for a given action, if present, cannot change within the lifetime of the fingerprint
            // cache (we call getKey with artifactExpander to check action key, when we are ready to
            // execute the action in case of a cache miss).
            commandLineItemMapFn.clearArtifactExpander();
          }
        } else {
          fingerprint.addInt(stringificationType.ordinal());
          actionKeyContext.addNestedSetToFingerprint(fingerprint, values);
        }
      } else {
        int count = (Integer) arguments.get(argi++);
        List<Object> maybeExpandedValues =
            maybeExpandDirectories(
                artifactExpander,
                arguments.subList(argi, argi + count),
                PathMappers.forActionKey(outputPathsMode));
        argi += count;
        if (mapEach != null) {
          // TODO(b/160181927): If artifactExpander == null (which happens in the analysis phase)
          // but expandDirectories is true, we run the map_each function on directory values without
          // actually expanding them. This differs from the real evaluation behavior. This means
          // that we can erroneously produce the same digest for two command lines that differ only
          // in their directory expansion. Fortunately, this is only a problem for shared action
          // conflict checking/aquery result, since at execution time we have an artifactExpander.
          applyMapEach(
              mapEach,
              maybeExpandedValues,
              fingerprint::addString,
              location,
              artifactExpander,
              PathMappers.forActionKey(outputPathsMode),
              starlarkSemantics);
        } else {
          for (Object value : maybeExpandedValues) {
            addSingleObjectToFingerprint(fingerprint, value);
          }
        }
      }
      if ((features & EXPAND_DIRECTORIES) != 0) {
        fingerprint.addUUID(EXPAND_DIRECTORIES_UUID);
      }
      if ((features & UNIQUIFY) != 0) {
        fingerprint.addUUID(UNIQUIFY_UUID);
      }
      if ((features & OMIT_IF_EMPTY) != 0) {
        fingerprint.addUUID(OMIT_IF_EMPTY_UUID);
      }
      if ((features & HAS_ARG_NAME) != 0) {
        String argName = (String) arguments.get(argi++);
        fingerprint.addUUID(ARG_NAME_UUID);
        fingerprint.addString(argName);
      }
      if ((features & HAS_FORMAT_EACH) != 0) {
        String formatStr = (String) arguments.get(argi++);
        fingerprint.addUUID(FORMAT_EACH_UUID);
        fingerprint.addString(formatStr);
      }
      if ((features & HAS_BEFORE_EACH) != 0) {
        String beforeEach = (String) arguments.get(argi++);
        fingerprint.addUUID(BEFORE_EACH_UUID);
        fingerprint.addString(beforeEach);
      } else if ((features & HAS_JOIN_WITH) != 0) {
        String joinWith = (String) arguments.get(argi++);
        fingerprint.addUUID(JOIN_WITH_UUID);
        fingerprint.addString(joinWith);
        if ((features & HAS_FORMAT_JOINED) != 0) {
          String formatJoined = (String) arguments.get(argi++);
          fingerprint.addUUID(FORMAT_JOINED_UUID);
          fingerprint.addString(formatJoined);
        }
      }
      if ((features & HAS_TERMINATE_WITH) != 0) {
        String terminateWith = (String) arguments.get(argi++);
        fingerprint.addUUID(TERMINATE_WITH_UUID);
        fingerprint.addString(terminateWith);
      }
      return argi;
    }

    static class Builder {
      @Nullable private final Sequence<?> list;
      @Nullable private final NestedSet<?> nestedSet;
      private final StringificationType nestedSetStringificationType;
      private Location location;
      public String argName;
      private boolean expandDirectories;
      private StarlarkCallable mapEach;
      private String formatEach;
      private String beforeEach;
      private String joinWith;
      private String formatJoined;
      private boolean omitIfEmpty;
      private boolean uniquify;
      private String terminateWith;

      Builder(Sequence<?> list) {
        this.list = list;
        this.nestedSet = null;
        this.nestedSetStringificationType = StringificationType.DEFAULT;
      }

      Builder(NestedSet<?> nestedSet, Class<?> nestedSetElementType) {
        this.list = null;
        this.nestedSet = nestedSet;
        if (nestedSetElementType == FileApi.class) {
          this.nestedSetStringificationType = StringificationType.FILE;
        } else if (nestedSetElementType == Label.class) {
          this.nestedSetStringificationType = StringificationType.LABEL;
        } else {
          this.nestedSetStringificationType = StringificationType.DEFAULT;
        }
      }

      @CanIgnoreReturnValue
      Builder setLocation(Location location) {
        this.location = location;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setArgName(String argName) {
        this.argName = argName;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setExpandDirectories(boolean expandDirectories) {
        this.expandDirectories = expandDirectories;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setMapEach(StarlarkCallable mapEach) {
        this.mapEach = mapEach;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setFormatEach(String format) {
        this.formatEach = format;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setBeforeEach(String beforeEach) {
        this.beforeEach = beforeEach;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setJoinWith(String joinWith) {
        this.joinWith = joinWith;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setFormatJoined(String formatJoined) {
        this.formatJoined = formatJoined;
        return this;
      }

      @CanIgnoreReturnValue
      Builder omitIfEmpty(boolean omitIfEmpty) {
        this.omitIfEmpty = omitIfEmpty;
        return this;
      }

      @CanIgnoreReturnValue
      Builder uniquify(boolean uniquify) {
        this.uniquify = uniquify;
        return this;
      }

      @CanIgnoreReturnValue
      Builder setTerminateWith(String terminateWith) {
        this.terminateWith = terminateWith;
        return this;
      }
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      VectorArg vectorArg = (VectorArg) o;
      return features == vectorArg.features
          && stringificationType.equals(vectorArg.stringificationType);
    }

    @Override
    public int hashCode() {
      return 31 * Integer.hashCode(features) + stringificationType.hashCode();
    }
  }

  @AutoCodec
  static final class ScalarArg {
    private static final Interner<ScalarArg> interner = BlazeInterners.newStrongInterner();
    private static final UUID FORMAT_UUID = UUID.fromString("8cb96642-a235-4fe0-b3ed-ebfdae8a0bd9");

    private final boolean hasFormat;

    private ScalarArg(boolean hasFormat) {
      this.hasFormat = hasFormat;
    }

    @VisibleForSerialization
    @AutoCodec.Instantiator
    static ScalarArg create(boolean hasFormat) {
      return interner.intern(new ScalarArg(hasFormat));
    }

    private static void push(List<Object> arguments, Builder arg) {
      ScalarArg scalarArg = ScalarArg.create(arg.format != null);
      arguments.add(scalarArg);
      arguments.add(arg.object);
      if (scalarArg.hasFormat) {
        arguments.add(arg.format);
      }
    }

    private int eval(
        List<Object> arguments,
        int argi,
        List<String> builder,
        PathMapper pathMapper,
        @Nullable RepositoryMapping mainRepoMapping) {
      Object object = arguments.get(argi++);
      String stringValue =
          StarlarkCustomCommandLine.expandToCommandLine(object, pathMapper, mainRepoMapping);
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        stringValue = SingleStringArgFormatter.format(formatStr, stringValue);
      }
      builder.add(stringValue);
      return argi;
    }

    private int addToFingerprint(List<Object> arguments, int argi, Fingerprint fingerprint) {
      Object object = arguments.get(argi++);
      addSingleObjectToFingerprint(fingerprint, object);
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        fingerprint.addUUID(FORMAT_UUID);
        fingerprint.addString(formatStr);
      }
      return argi;
    }

    static class Builder {
      private final Object object;
      private String format;

      Builder(Object object) {
        this.object = object;
      }

      @CanIgnoreReturnValue
      Builder setFormat(String format) {
        this.format = format;
        return this;
      }
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      ScalarArg scalarArg = (ScalarArg) o;
      return hasFormat == scalarArg.hasFormat;
    }

    @Override
    public int hashCode() {
      return Boolean.hashCode(hasFormat);
    }
  }

  static final class Builder {
    private final StarlarkSemantics starlarkSemantics;
    private final List<Object> arguments = new ArrayList<>();
    // Indexes in arguments list where individual args begin
    private final ImmutableList.Builder<Integer> argStartIndexes = ImmutableList.builder();

    public Builder(StarlarkSemantics starlarkSemantics) {
      this.starlarkSemantics = starlarkSemantics;
    }

    @CanIgnoreReturnValue
    Builder recordArgStart() {
      argStartIndexes.add(arguments.size());
      return this;
    }

    @CanIgnoreReturnValue
    Builder add(Object object) {
      arguments.add(object);
      return this;
    }

    @CanIgnoreReturnValue
    Builder add(VectorArg.Builder vectorArg) {
      VectorArg.push(arguments, vectorArg, starlarkSemantics);
      return this;
    }

    @CanIgnoreReturnValue
    Builder add(ScalarArg.Builder scalarArg) {
      ScalarArg.push(arguments, scalarArg);
      return this;
    }

    StarlarkCustomCommandLine build(
        boolean flagPerLine, @Nullable RepositoryMapping mainRepoMapping) {
      Object[] args;
      if (mainRepoMapping != null) {
        args = arguments.toArray(new Object[arguments.size() + 1]);
        args[arguments.size()] = mainRepoMapping;
      } else {
        args = arguments.toArray();
      }
      return flagPerLine
          ? new StarlarkCustomCommandLineWithIndexes(args, argStartIndexes.build())
          : new StarlarkCustomCommandLine(args);
    }
  }

  /**
   * Stored as an {@code Object[]} instead of an {@link ImmutableList} to save memory, but is never
   * modified. Access via {@link #rawArgsAsList} for an unmodifiable {@link List} view.
   */
  private final Object[] arguments;

  private StarlarkCustomCommandLine(Object[] arguments) {
    this.arguments = arguments;
  }

  /** Wraps {@link #arguments} in an unmodifiable {@link List} view. */
  private List<Object> rawArgsAsList() {
    return Collections.unmodifiableList(Arrays.asList(arguments));
  }

  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException, InterruptedException {
    return arguments(null, PathMapper.NOOP);
  }

  @Override
  public Iterable<String> arguments(
      @Nullable ArtifactExpander artifactExpander, PathMapper pathMapper)
      throws CommandLineExpansionException, InterruptedException {
    List<String> result = new ArrayList<>();
    List<Object> arguments = rawArgsAsList();

    RepositoryMapping mainRepoMapping;
    int size;
    // Added in #build() if any labels in the command line require this to be formatted with an
    // apparent repository name.
    if (!arguments.isEmpty() && arguments.getLast() instanceof RepositoryMapping) {
      mainRepoMapping = (RepositoryMapping) arguments.getLast();
      size = arguments.size() - 1;
    } else {
      mainRepoMapping = null;
      size = arguments.size();
    }

    for (int argi = 0; argi < size; ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi =
            ((VectorArg) arg)
                .eval(arguments, argi, result, artifactExpander, pathMapper, mainRepoMapping);

      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).eval(arguments, argi, result, pathMapper, mainRepoMapping);
      } else {
        result.add(expandToCommandLine(arg, pathMapper, mainRepoMapping));
      }
    }
    return ImmutableList.copyOf(pathMapper.mapCustomStarlarkArgs(result));
  }

  private static String expandToCommandLine(
      Object object, PathMapper pathMapper, @Nullable RepositoryMapping mainRepoMapping) {
    if (mainRepoMapping != null && object instanceof Label label) {
      return label.getDisplayForm(mainRepoMapping);
    }

    // It'd be nice to build this into DerivedArtifact's CommandLine interface so we don't have
    // to explicitly check if an object is a DerivedArtifact. Unfortunately that would require
    // a lot more dependencies on the Java library DerivedArtifact is built into.
    return object instanceof DerivedArtifact
        ? pathMapper.map(((DerivedArtifact) object).getExecPath()).getPathString()
        : CommandLineItem.expandToCommandLine(object);
  }

  private static void addSingleObjectToFingerprint(Fingerprint fingerprint, Object object) {
    StringificationType stringificationType =
        switch (object) {
          case FileApi ignored -> StringificationType.FILE;
          case Label ignored -> StringificationType.LABEL;
          default -> StringificationType.DEFAULT;
        };
    fingerprint.addInt(stringificationType.ordinal());
    fingerprint.addString(CommandLineItem.expandToCommandLine(object));
  }

  private static class StarlarkCustomCommandLineWithIndexes extends StarlarkCustomCommandLine {
    /**
     * If non-empty, an extra level of grouping on top of the 'arguments' list. Each element is the
     * beginning of a group of args. For example, if this contains 0 and 3, then arguments 0, 1 and
     * 2 constitute the first group, and arguments 3 to the end constitute the next. The expanded
     * version of these arguments will be concatenated together to support flag_per_line format.
     */
    private final ImmutableList<Integer> argStartIndexes;

    StarlarkCustomCommandLineWithIndexes(
        Object[] arguments, ImmutableList<Integer> argStartIndexes) {
      super(arguments);
      this.argStartIndexes = argStartIndexes;
    }

    @Override
    public Iterable<String> arguments(
        @Nullable ArtifactExpander artifactExpander, PathMapper pathMapper)
        throws CommandLineExpansionException, InterruptedException {

      List<String> result = new ArrayList<>();
      List<Object> arguments = ((StarlarkCustomCommandLine) this).rawArgsAsList();

      // If we're grouping arguments, keep track of the result indexes corresponding to the
      // argStartIndexes, reflecting VectorArg and ScalarArg expansion.
      List<Integer> resultGroupStarts =
          argStartIndexes.isEmpty() ? ImmutableList.of() : new ArrayList<>();
      Iterator<Integer> startIndexIterator = argStartIndexes.iterator();
      int nextStartIndex = startIndexIterator.hasNext() ? startIndexIterator.next() : -1;

      RepositoryMapping mainRepoMapping;
      int size;
      if (!arguments.isEmpty() && arguments.getLast() instanceof RepositoryMapping) {
        mainRepoMapping = (RepositoryMapping) arguments.getLast();
        size = arguments.size() - 1;
      } else {
        mainRepoMapping = null;
        size = arguments.size();
      }

      for (int argi = 0; argi < size; ) {

        // If we're grouping arguments, record the actual beginning of each group
        if (argi == nextStartIndex) {
          resultGroupStarts.add(result.size());
          nextStartIndex = startIndexIterator.hasNext() ? startIndexIterator.next() : -1;
        }

        Object arg = arguments.get(argi++);
        if (arg instanceof VectorArg) {
          argi =
              ((VectorArg) arg)
                  .eval(arguments, argi, result, artifactExpander, pathMapper, mainRepoMapping);
        } else if (arg instanceof ScalarArg) {
          argi = ((ScalarArg) arg).eval(arguments, argi, result, pathMapper, mainRepoMapping);
        } else {
          result.add(
              StarlarkCustomCommandLine.expandToCommandLine(arg, pathMapper, mainRepoMapping));
        }
      }

      if (argStartIndexes.isEmpty()) {
        // Normal case, no further grouping
        return ImmutableList.copyOf(result);
      }

      // Grouped case -- concatenate results.
      ImmutableList.Builder<String> groupedBuilder = ImmutableList.builder();
      int numStarts = resultGroupStarts.size();
      resultGroupStarts.add(result.size());
      for (int i = 0; i < numStarts; i++) {
        // Arguments that constitute a single group
        List<String> group = result.subList(resultGroupStarts.get(i), resultGroupStarts.get(i + 1));
        if (group.size() < 2) {
          groupedBuilder.addAll(group);
        } else {
          // "--x=y z", or just "y z"
          String first = group.get(0);
          String rest = String.join(" ", group.subList(1, group.size()));
          groupedBuilder.add(first.isEmpty() ? rest : (first + '=' + rest));
        }
      }
      return groupedBuilder.build();
    }
  }

  @Override
  public void addToFingerprint(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      CoreOptions.OutputPathsMode outputPathsMode,
      Fingerprint fingerprint)
      throws CommandLineExpansionException, InterruptedException {
    List<Object> arguments = rawArgsAsList();
    int size;
    if (!arguments.isEmpty() && arguments.getLast() instanceof RepositoryMapping mainRepoMapping) {
      fingerprint.addStringMap(
          Maps.transformValues(mainRepoMapping.entries(), RepositoryName::getName));
      size = arguments.size() - 1;
    } else {
      size = arguments.size();
    }
    for (int argi = 0; argi < size; ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi =
            ((VectorArg) arg)
                .addToFingerprint(
                    arguments,
                    argi,
                    actionKeyContext,
                    fingerprint,
                    artifactExpander,
                    outputPathsMode);
      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).addToFingerprint(arguments, argi, fingerprint);
      } else {
        addSingleObjectToFingerprint(fingerprint, arg);
      }
    }
  }

  /** Used during action key evaluation when we don't have an artifact expander. */
  private static class NoopExpander implements DirectoryExpander {
    @Override
    public ImmutableList<FileApi> list(FileApi file) {
      return ImmutableList.of(file);
    }

    static final DirectoryExpander INSTANCE = new NoopExpander();
  }

  private static final class FullExpander implements DirectoryExpander {
    private final ArtifactExpander expander;

    FullExpander(ArtifactExpander expander) {
      this.expander = expander;
    }

    @Override
    public ImmutableList<FileApi> list(FileApi file) {
      Artifact artifact = (Artifact) file;
      if (artifact.isTreeArtifact()) {
        List<Artifact> files = new ArrayList<>(1);
        expander.expand((Artifact) file, files);
        return ImmutableList.copyOf(files);
      } else {
        return ImmutableList.of(file);
      }
    }
  }

  private static void applyMapEach(
      StarlarkCallable mapFn,
      List<Object> originalValues,
      Consumer<String> consumer,
      Location loc,
      @Nullable ArtifactExpander artifactExpander,
      PathMapper pathMapper,
      StarlarkSemantics starlarkSemantics)
      throws CommandLineExpansionException, InterruptedException {
    try (Mutability mu = Mutability.create("map_each")) {
      StarlarkThread thread = new StarlarkThread(mu, pathMapper.storeIn(starlarkSemantics));
      // TODO(b/77140311): Error if we issue print statements.
      thread.setPrintHandler((th, msg) -> {});
      int count = originalValues.size();
      // map_each can accept either each object, or each object + a directory expander.
      boolean wantsDirectoryExpander =
          (mapFn instanceof StarlarkFunction)
              && ((StarlarkFunction) mapFn).getParameterNames().size() >= 2;
      // We create a list that we reuse for the args to map_each
      List<Object> args = new ArrayList<>(2);
      args.add(null); // This will be overwritten each iteration.
      if (wantsDirectoryExpander) {
        final DirectoryExpander expander;
        if (artifactExpander != null) {
          expander = new FullExpander(artifactExpander);
        } else {
          expander = NoopExpander.INSTANCE;
        }
        args.add(expander); // This will remain constant each iteration
      }
      for (int i = 0; i < count; ++i) {
        args.set(0, originalValues.get(i));
        Object ret = Starlark.call(thread, mapFn, args, /* kwargs= */ ImmutableMap.of());
        if (ret instanceof String) {
          consumer.accept((String) ret);
        } else if (ret instanceof Sequence) {
          for (Object val : ((Sequence<?>) ret)) {
            if (!(val instanceof String)) {
              throw new CommandLineExpansionException(
                  "Expected map_each to return string, None, or list of strings, "
                      + "found list containing "
                      + Starlark.type(val));
            }
            consumer.accept((String) val);
          }
        } else if (ret != Starlark.NONE) {
          throw new CommandLineExpansionException(
              "Expected map_each to return string, None, or list of strings, found "
                  + Starlark.type(ret));
        }
      }
    } catch (EvalException e) {
      // TODO(adonovan): consider calling a wrapper function to interpose a fake stack
      // frame that establishes the args.add_all call at loc. Or manipulating the stack
      // before printing it.
      throw new CommandLineExpansionException(
          errorMessage(e.getMessageWithStack(), loc, e.getCause()));
    }
  }

  private static class CommandLineItemMapEachAdaptor
      extends CommandLineItem.ParametrizedMapFn<Object> {
    private final StarlarkCallable mapFn;
    private final Location location;
    private final StarlarkSemantics starlarkSemantics;

    /**
     * Indicates whether artifactExpander was provided on construction. This is used to distinguish
     * the case where it's not provided from the case where it was provided but subsequently
     * cleared.
     */
    private final boolean hasArtifactExpander;

    private final CoreOptions.OutputPathsMode outputPathsMode;

    @Nullable private ArtifactExpander artifactExpander;

    CommandLineItemMapEachAdaptor(
        StarlarkCallable mapFn,
        Location location,
        StarlarkSemantics starlarkSemantics,
        @Nullable ArtifactExpander artifactExpander,
        CoreOptions.OutputPathsMode outputPathsMode) {
      this.mapFn = mapFn;
      this.location = location;
      this.starlarkSemantics = starlarkSemantics;
      this.hasArtifactExpander = artifactExpander != null;
      this.artifactExpander = artifactExpander;
      this.outputPathsMode = outputPathsMode;
    }

    @Override
    public void expandToCommandLine(Object object, Consumer<String> args)
        throws CommandLineExpansionException, InterruptedException {
      Preconditions.checkState(artifactExpander != null || !hasArtifactExpander);
      applyMapEach(
          mapFn,
          maybeExpandDirectory(object),
          args,
          location,
          artifactExpander,
          PathMappers.forActionKey(outputPathsMode),
          starlarkSemantics);
    }

    private List<Object> maybeExpandDirectory(Object object) throws CommandLineExpansionException {
      if (artifactExpander == null || !VectorArg.isDirectory(object)) {
        return ImmutableList.of(object);
      }

      return VectorArg.expandDirectories(
          artifactExpander, ImmutableList.of(object), PathMappers.forActionKey(outputPathsMode));
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof CommandLineItemMapEachAdaptor)) {
        return false;
      }
      CommandLineItemMapEachAdaptor other = (CommandLineItemMapEachAdaptor) obj;
      // Instance compare intentional
      // The normal implementation uses location + name of function,
      // which can conceivably conflict in tests
      // We only compare presence of artifactExpander vs absence of it since the nestedset
      // fingerprint cache is emptied after every build, therefore if the artifact expander is
      // provided, it will be the same.
      return mapFn == other.mapFn
          && hasArtifactExpander == other.hasArtifactExpander
          && outputPathsMode == other.outputPathsMode;
    }

    @Override
    public int hashCode() {
      // Force use of identityHashCode, in case the callable uses a custom hash function. (As of
      // this writing, only providers seem to have a custom hashCode, and those shouldn't be used
      // as map_each functions, but doesn't hurt to be safe...).
      return outputPathsMode.hashCode()
          + 31
              * (Boolean.hashCode(hasArtifactExpander) + 31 * (System.identityHashCode(mapFn) + 1));
    }

    @Override
    public int maxInstancesAllowed() {
      // No limit to these, as this is just a wrapper for Starlark functions, which are
      // always static
      return Integer.MAX_VALUE;
    }

    /**
     * Clears the artifact expander in order not to prolong the lifetime of it unnecessarily.
     *
     * <p>Although this operation technically changes this object, it can be called after we add the
     * object to a {@link HashSet}. Clearing the artifactExpander does not affect the result of
     * {@link #equals} or {@link #hashCode}. Please note that once we call this function, we can no
     * longer call {@link #expandToCommandLine}.
     */
    void clearArtifactExpander() {
      artifactExpander = null;
    }
  }

  private static String errorMessage(
      String message, @Nullable Location location, @Nullable Throwable cause) {
    return LINE_JOINER.join(
        "\n", FIELD_JOINER.join(location, message), getCauseMessage(cause, message));
  }

  @Nullable
  private static String getCauseMessage(@Nullable Throwable cause, String message) {
    if (cause == null) {
      return null;
    }
    String causeMessage = cause.getMessage();
    if (causeMessage == null) {
      return null;
    }
    if (message == null) {
      return causeMessage;
    }
    // Skip the cause if it is redundant with the message so far.
    if (message.contains(causeMessage)) {
      return null;
    }
    return causeMessage;
  }

  /**
   * When we expand filesets the user might still expect a File object (since the results may be fed
   * into map_each. Therefore we synthesize a File object from the fileset symlink.
   */
  static class FilesetSymlinkFile implements FileApi, CommandLineItem {
    private final Artifact fileset;
    private final PathFragment execPath;

    FilesetSymlinkFile(Artifact fileset, PathFragment execPath) {
      this.fileset = fileset;
      this.execPath = execPath;
    }

    @Override
    public String getDirnameForStarlark(StarlarkSemantics semantics) {
      PathFragment parent = execPath.getParentDirectory();
      return (parent == null) ? "/" : parent.getSafePathString();
    }

    @Override
    public String getFilename() {
      return execPath.getBaseName();
    }

    @Override
    public String getExtension() {
      return execPath.getFileExtension();
    }

    @Override
    public Label getOwnerLabel() {
      return fileset.getOwnerLabel();
    }

    @Override
    public FileRootApi getRootForStarlark(StarlarkSemantics semantics) {
      return fileset.getRoot();
    }

    @Override
    public boolean isSourceArtifact() {
      // This information is lost to us.
      // Since the symlinks are always in the output tree, settle for saying "no"
      return false;
    }

    @Override
    public boolean isDirectory() {
      return false;
    }

    @Override
    public String getRunfilesPathString() {
      PathFragment relativePath = execPath.relativeTo(fileset.getExecPath());
      return fileset.getRunfilesPath().getRelative(relativePath).getPathString();
    }

    @Override
    public String getExecPathStringForStarlark(StarlarkSemantics semantics) {
      return execPath.getPathString();
    }

    @Override
    public String getTreeRelativePathString() throws EvalException {
      throw Starlark.errorf(
          "tree_relative_path not allowed for files that are not tree artifact files.");
    }

    @Override
    public String expandToCommandLine() {
      return execPath.getPathString();
    }

    @Override
    public void repr(Printer printer) {
      if (isSourceArtifact()) {
        printer.append("<source file " + getRunfilesPathString() + ">");
      } else {
        printer.append("<generated file " + getRunfilesPathString() + ">");
      }
    }
  }
}
