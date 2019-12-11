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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.actions.SingleStringArgFormatter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Supports ctx.actions.args() from Skylark. */
@AutoCodec
public class SkylarkCustomCommandLine extends CommandLine {
  private final StarlarkSemantics starlarkSemantics;
  private final ImmutableList<Object> arguments;

  private static final Joiner LINE_JOINER = Joiner.on("\n").skipNulls();
  private static final Joiner FIELD_JOINER = Joiner.on(": ").skipNulls();

  @AutoCodec
  static final class VectorArg {
    private static final Interner<VectorArg> interner = BlazeInterners.newStrongInterner();

    private static final int HAS_LOCATION = 1;
    private static final int HAS_MAP_ALL = 1 << 1;
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

    private VectorArg(int features) {
      this.features = features;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static VectorArg create(int features) {
      return interner.intern(new VectorArg(features));
    }

    private static void push(ImmutableList.Builder<Object> arguments, Builder arg) {
      int features = 0;
      features |= arg.mapAll != null ? HAS_MAP_ALL : 0;
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
              && (features & (HAS_FORMAT_EACH | HAS_FORMAT_JOINED | HAS_MAP_ALL | HAS_MAP_EACH))
                  != 0;
      features |= hasLocation ? HAS_LOCATION : 0;
      Preconditions.checkState(
          (features & (HAS_MAP_ALL | HAS_MAP_EACH)) != (HAS_MAP_ALL | HAS_MAP_EACH),
          "Cannot use both map_all and map_each");
      VectorArg vectorArg = VectorArg.create(features);
      arguments.add(vectorArg);
      if (hasLocation) {
        arguments.add(arg.location);
      }
      if (arg.mapAll != null) {
        arguments.add(arg.mapAll);
      }
      if (arg.mapEach != null) {
        arguments.add(arg.mapEach);
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
        ImmutableList.Builder<String> builder,
        @Nullable ArtifactExpander artifactExpander,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      final Location location =
          ((features & HAS_LOCATION) != 0) ? (Location) arguments.get(argi++) : null;
      final List<Object> originalValues;
      BaseFunction mapAll =
          ((features & HAS_MAP_ALL) != 0) ? (BaseFunction) arguments.get(argi++) : null;
      BaseFunction mapEach =
          ((features & HAS_MAP_EACH) != 0) ? (BaseFunction) arguments.get(argi++) : null;
      if ((features & IS_NESTED_SET) != 0) {
        @SuppressWarnings("unchecked")
        NestedSet<Object> nestedSet = (NestedSet<Object>) arguments.get(argi++);
        originalValues = nestedSet.toList();
      } else {
        int count = (Integer) arguments.get(argi++);
        originalValues = arguments.subList(argi, argi + count);
        argi += count;
      }
      List<Object> expandedValues = originalValues;
      if (artifactExpander != null && (features & EXPAND_DIRECTORIES) != 0) {
        if (hasDirectory(originalValues)) {
          expandedValues = expandDirectories(artifactExpander, originalValues);
        }
      }
      List<String> stringValues;
      if (mapEach != null) {
        stringValues = new ArrayList<>(expandedValues.size());
        applyMapEach(mapEach, expandedValues, stringValues::add, location, starlarkSemantics);
      } else if (mapAll != null) {
        Object result = applyMapFn(mapAll, expandedValues, location, starlarkSemantics);
        if (!(result instanceof List)) {
          throw new CommandLineExpansionException(
              errorMessage(
                  "map_fn must return a list, got " + result.getClass().getSimpleName(),
                  location,
                  null));
        }
        List<?> resultAsList = (List) result;
        if (resultAsList.size() != expandedValues.size()) {
          throw new CommandLineExpansionException(
              errorMessage(
                  String.format(
                      "map_fn must return a list of the same length as the input. "
                          + "Found list of length %d, expected %d.",
                      resultAsList.size(), expandedValues.size()),
                  location,
                  null));
        }
        int count = resultAsList.size();
        stringValues = new ArrayList<>(count);
        // map_fn contract doesn't guarantee that the values returned are strings,
        // so convert here
        for (int i = 0; i < count; ++i) {
          stringValues.add(CommandLineItem.expandToCommandLine(resultAsList.get(i)));
        }
      } else {
        int count = expandedValues.size();
        stringValues = new ArrayList<>(expandedValues.size());
        for (int i = 0; i < count; ++i) {
          stringValues.add(CommandLineItem.expandToCommandLine(expandedValues.get(i)));
        }
      }
      // It's safe to uniquify at this stage, any transformations after this
      // will ensure continued uniqueness of the values
      if ((features & UNIQUIFY) != 0) {
        HashSet<String> seen = new HashSet<>(stringValues.size());
        int count = stringValues.size();
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
      return ((object instanceof Artifact) && ((Artifact) object).isDirectory());
    }

    private static List<Object> expandDirectories(
        Artifact.ArtifactExpander artifactExpander, List<Object> originalValues)
        throws CommandLineExpansionException {
      List<Object> expandedValues;
      int n = originalValues.size();
      expandedValues = new ArrayList<>(n);
      for (int i = 0; i < n; ++i) {
        Object object = originalValues.get(i);
        if (isDirectory(object)) {
          Artifact artifact = (Artifact) object;
          if (artifact.isTreeArtifact()) {
            artifactExpander.expand((Artifact) object, expandedValues);
          } else if (artifact.isFileset()) {
            expandFileset(artifactExpander, artifact, expandedValues);
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
        Artifact.ArtifactExpander artifactExpander, Artifact fileset, List<Object> expandedValues)
        throws CommandLineExpansionException {
      try {
        FilesetManifest filesetManifest =
            FilesetManifest.constructFilesetManifest(
                artifactExpander.getFileset(fileset),
                fileset.getExecPath(),
                RelativeSymlinkBehavior.IGNORE);
        for (PathFragment relativePath : filesetManifest.getEntries().keySet()) {
          expandedValues.add(new FilesetSymlinkFile(fileset, relativePath));
        }
      } catch (IOException e) {
        throw new CommandLineExpansionException("Could not expand fileset: " + e.getMessage());
      }
    }

    private int addToFingerprint(
        List<Object> arguments,
        int argi,
        ActionKeyContext actionKeyContext,
        Fingerprint fingerprint,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      if ((features & HAS_MAP_ALL) != 0) {
        return addToFingerprintLegacy(arguments, argi, fingerprint, starlarkSemantics);
      }
      final Location location =
          ((features & HAS_LOCATION) != 0) ? (Location) arguments.get(argi++) : null;
      BaseFunction mapEach =
          ((features & HAS_MAP_EACH) != 0) ? (BaseFunction) arguments.get(argi++) : null;
      if ((features & IS_NESTED_SET) != 0) {
        NestedSet<?> values = (NestedSet) arguments.get(argi++);
        if (mapEach != null) {
          CommandLineItem.MapFn<Object> commandLineItemMapFn =
              new CommandLineItemMapEachAdaptor(mapEach, location, starlarkSemantics);
          try {
            actionKeyContext.addNestedSetToFingerprint(commandLineItemMapFn, fingerprint, values);
          } catch (UncheckedCommandLineExpansionException e) {
            // We wrap the CommandLineExpansionException below, unwrap here
            throw e.cause;
          }
        } else {
          actionKeyContext.addNestedSetToFingerprint(fingerprint, values);
        }
      } else {
        int count = (Integer) arguments.get(argi++);
        final List<Object> originalValues = arguments.subList(argi, argi + count);
        argi += count;
        if (mapEach != null) {
          List<String> stringValues = new ArrayList<>(count);
          applyMapEach(mapEach, originalValues, stringValues::add, location, starlarkSemantics);
          for (String s : stringValues) {
            fingerprint.addString(s);
          }
        } else {
          for (int i = 0; i < count; ++i) {
            fingerprint.addString(CommandLineItem.expandToCommandLine(originalValues.get(i)));
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

    private int addToFingerprintLegacy(
        List<Object> arguments,
        int argi,
        Fingerprint fingerprint,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      ImmutableList.Builder<String> builder = ImmutableList.builder();
      argi = eval(arguments, argi, builder, null, starlarkSemantics);
      for (String s : builder.build()) {
        fingerprint.addString(s);
      }
      return argi;
    }

    static class Builder {
      @Nullable private final Sequence<?> list;
      @Nullable private final NestedSet<?> nestedSet;
      private Location location;
      public String argName;
      private boolean expandDirectories;
      private BaseFunction mapAll;
      private BaseFunction mapEach;
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
      }

      Builder(NestedSet<?> nestedSet) {
        this.list = null;
        this.nestedSet = nestedSet;
      }

      Builder setLocation(Location location) {
        this.location = location;
        return this;
      }

      Builder setArgName(String argName) {
        this.argName = argName;
        return this;
      }

      Builder setExpandDirectories(boolean expandDirectories) {
        this.expandDirectories = expandDirectories;
        return this;
      }

      Builder setMapAll(BaseFunction mapAll) {
        this.mapAll = mapAll;
        return this;
      }

      Builder setMapEach(BaseFunction mapEach) {
        this.mapEach = mapEach;
        return this;
      }

      Builder setFormatEach(String format) {
        this.formatEach = format;
        return this;
      }

      Builder setBeforeEach(String beforeEach) {
        this.beforeEach = beforeEach;
        return this;
      }

      Builder setJoinWith(String joinWith) {
        this.joinWith = joinWith;
        return this;
      }

      Builder setFormatJoined(String formatJoined) {
        this.formatJoined = formatJoined;
        return this;
      }

      Builder omitIfEmpty(boolean omitIfEmpty) {
        this.omitIfEmpty = omitIfEmpty;
        return this;
      }

      Builder uniquify(boolean uniquify) {
        this.uniquify = uniquify;
        return this;
      }

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
      return features == vectorArg.features;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(features);
    }
  }

  @AutoCodec
  static final class ScalarArg {
    private static final Interner<ScalarArg> interner = BlazeInterners.newStrongInterner();
    private static final UUID FORMAT_UUID = UUID.fromString("8cb96642-a235-4fe0-b3ed-ebfdae8a0bd9");

    private final boolean hasFormat;
    private final boolean hasMapFn;
    private final boolean hasLocation;

    private ScalarArg(boolean hasFormat, boolean hasMapFn, boolean hasLocation) {
      this.hasFormat = hasFormat;
      this.hasMapFn = hasMapFn;
      this.hasLocation = hasLocation;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static ScalarArg create(boolean hasFormat, boolean hasMapFn, boolean hasLocation) {
      return interner.intern(new ScalarArg(hasFormat, hasMapFn, hasLocation));
    }

    private static void push(ImmutableList.Builder<Object> arguments, Builder arg) {
      boolean wantsLocation = arg.format != null || arg.mapFn != null;
      boolean hasLocation = arg.location != null && wantsLocation;
      ScalarArg scalarArg = ScalarArg.create(arg.format != null, arg.mapFn != null, hasLocation);
      arguments.add(scalarArg);
      arguments.add(arg.object);
      if (hasLocation) {
        arguments.add(arg.location);
      }
      if (scalarArg.hasMapFn) {
        arguments.add(arg.mapFn);
      }
      if (scalarArg.hasFormat) {
        arguments.add(arg.format);
      }
    }

    private int eval(
        List<Object> arguments,
        int argi,
        ImmutableList.Builder<String> builder,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      Object object = arguments.get(argi++);
      final Location location = hasLocation ? (Location) arguments.get(argi++) : null;
      if (hasMapFn) {
        BaseFunction mapFn = (BaseFunction) arguments.get(argi++);
        object = applyMapFn(mapFn, object, location, starlarkSemantics);
      }
      String stringValue = CommandLineItem.expandToCommandLine(object);
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        stringValue = SingleStringArgFormatter.format(formatStr, stringValue);
      }
      builder.add(stringValue);
      return argi;
    }

    private int addToFingerprint(
        List<Object> arguments,
        int argi,
        Fingerprint fingerprint,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      if (hasMapFn) {
        return addToFingerprintLegacy(arguments, argi, fingerprint, starlarkSemantics);
      }
      Object object = arguments.get(argi++);
      String stringValue = CommandLineItem.expandToCommandLine(object);
      fingerprint.addString(stringValue);
      if (hasLocation) {
        argi++; // Skip past location slot
      }
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        fingerprint.addUUID(FORMAT_UUID);
        fingerprint.addString(formatStr);
      }
      return argi;
    }

    private int addToFingerprintLegacy(
        List<Object> arguments,
        int argi,
        Fingerprint fingerprint,
        StarlarkSemantics starlarkSemantics)
        throws CommandLineExpansionException {
      ImmutableList.Builder<String> builder = ImmutableList.builderWithExpectedSize(1);
      argi = eval(arguments, argi, builder, starlarkSemantics);
      for (String s : builder.build()) {
        fingerprint.addString(s);
      }
      return argi;
    }

    static class Builder {
      private Object object;
      private String format;
      private BaseFunction mapFn;
      private Location location;

      Builder(Object object) {
        this.object = object;
      }

      Builder setLocation(Location location) {
        this.location = location;
        return this;
      }

      Builder setFormat(String format) {
        this.format = format;
        return this;
      }

      Builder setMapFn(BaseFunction mapFn) {
        this.mapFn = mapFn;
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
      return hasFormat == scalarArg.hasFormat
          && hasMapFn == scalarArg.hasMapFn
          && hasLocation == scalarArg.hasLocation;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(hasFormat, hasMapFn, hasLocation);
    }
  }

  static class Builder {
    private final StarlarkSemantics starlarkSemantics;
    private final ImmutableList.Builder<Object> arguments = ImmutableList.builder();

    public Builder(StarlarkSemantics starlarkSemantics) {
      this.starlarkSemantics = starlarkSemantics;
    }

    Builder add(Object object) {
      arguments.add(object);
      return this;
    }

    Builder add(VectorArg.Builder vectorArg) {
      VectorArg.push(arguments, vectorArg);
      return this;
    }

    Builder add(ScalarArg.Builder scalarArg) {
      ScalarArg.push(arguments, scalarArg);
      return this;
    }

    SkylarkCustomCommandLine build() {
      return new SkylarkCustomCommandLine(starlarkSemantics, arguments.build());
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  SkylarkCustomCommandLine(StarlarkSemantics starlarkSemantics, ImmutableList<Object> arguments) {
    this.arguments = arguments;
    this.starlarkSemantics = starlarkSemantics;
  }

  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException {
    return arguments(null);
  }

  @Override
  public Iterable<String> arguments(@Nullable ArtifactExpander artifactExpander)
      throws CommandLineExpansionException {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (int argi = 0; argi < arguments.size(); ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi = ((VectorArg) arg).eval(arguments, argi, result, artifactExpander, starlarkSemantics);
      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).eval(arguments, argi, result, starlarkSemantics);
      } else {
        result.add(CommandLineItem.expandToCommandLine(arg));
      }
    }
    return result.build();
  }

  @Override
  public void addToFingerprint(ActionKeyContext actionKeyContext, Fingerprint fingerprint)
      throws CommandLineExpansionException {
    for (int argi = 0; argi < arguments.size(); ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi =
            ((VectorArg) arg)
                .addToFingerprint(
                    arguments, argi, actionKeyContext, fingerprint, starlarkSemantics);
      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).addToFingerprint(arguments, argi, fingerprint, starlarkSemantics);
      } else {
        fingerprint.addString(CommandLineItem.expandToCommandLine(arg));
      }
    }
  }

  private static Object applyMapFn(
      BaseFunction mapFn, Object arg, Location location, StarlarkSemantics starlarkSemantics)
      throws CommandLineExpansionException {
    ImmutableList<Object> args = ImmutableList.of(arg);
    try (Mutability mutability = Mutability.create("map_fn")) {
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setSemantics(starlarkSemantics)
              .setEventHandler(NullEventHandler.INSTANCE)
              .build();
      return Starlark.call(thread, mapFn, /*call=*/ null, args, /*kwargs=*/ ImmutableMap.of());
    } catch (EvalException e) {
      throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, e.getCause()));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new CommandLineExpansionException(
          errorMessage("Thread was interrupted", location, null));
    }
  }

  private static void applyMapEach(
      BaseFunction mapFn,
      List<Object> originalValues,
      Consumer<String> consumer,
      Location location,
      StarlarkSemantics starlarkSemantics)
      throws CommandLineExpansionException {
    try (Mutability mutability = Mutability.create("map_each")) {
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setSemantics(starlarkSemantics)
              // TODO(b/77140311): Error if we issue print statements
              .setEventHandler(NullEventHandler.INSTANCE)
              .build();
      int count = originalValues.size();
      for (int i = 0; i < count; ++i) {
        Object ret =
            Starlark.call(
                thread,
                mapFn,
                /*call=*/ null,
                originalValues.subList(i, i + 1),
                /*kwargs=*/ ImmutableMap.of());
        if (ret instanceof String) {
          consumer.accept((String) ret);
        } else if (ret instanceof Sequence) {
          for (Object val : ((Sequence) ret)) {
            if (!(val instanceof String)) {
              throw new CommandLineExpansionException(
                  "Expected map_each to return string, None, or list of strings, "
                      + "found list containing "
                      + val.getClass().getSimpleName());
            }
            consumer.accept((String) val);
          }
        } else if (ret != Starlark.NONE) {
          throw new CommandLineExpansionException(
              "Expected map_each to return string, None, or list of strings, found "
                  + ret.getClass().getSimpleName());
        }
      }
    } catch (EvalException e) {
      throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, e.getCause()));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new CommandLineExpansionException(
          errorMessage("Thread was interrupted", location, null));
    }
  }

  private static class CommandLineItemMapEachAdaptor
      extends CommandLineItem.ParametrizedMapFn<Object> {
    private final BaseFunction mapFn;
    private final Location location;
    private final StarlarkSemantics starlarkSemantics;

    CommandLineItemMapEachAdaptor(
        BaseFunction mapFn, Location location, StarlarkSemantics starlarkSemantics) {
      this.mapFn = mapFn;
      this.location = location;
      this.starlarkSemantics = starlarkSemantics;
    }

    @Override
    public void expandToCommandLine(Object object, Consumer<String> args) {
      try {
        applyMapEach(mapFn, ImmutableList.of(object), args, location, starlarkSemantics);
      } catch (CommandLineExpansionException e) {
        // Rather than update CommandLineItem#expandToCommandLine and the numerous callers,
        // we wrap this in a runtime exception and handle it above
        throw new UncheckedCommandLineExpansionException(e);
      }
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
      return mapFn == other.mapFn;
    }

    @Override
    public int hashCode() {
      // identity hashcode intentional
      return System.identityHashCode(mapFn);
    }

    @Override
    public int maxInstancesAllowed() {
      // No limit to these, as this is just a wrapper for Skylark functions, which are
      // always static
      return Integer.MAX_VALUE;
    }
  }

  private static String errorMessage(
      String message, @Nullable Location location, @Nullable Throwable cause) {
    return LINE_JOINER.join(
        "\n", FIELD_JOINER.join(location, message), getCauseMessage(cause, message));
  }

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

  private static class UncheckedCommandLineExpansionException extends RuntimeException {
    final CommandLineExpansionException cause;

    UncheckedCommandLineExpansionException(CommandLineExpansionException cause) {
      this.cause = cause;
    }
  }

  /**
   * When we expand filesets the user might still expect a File object (since the results may be fed
   * into map_each. Therefore we synthesize a File object from the fileset symlink.
   */
  static class FilesetSymlinkFile implements FileApi, CommandLineItem {
    private final Artifact fileset;
    private final PathFragment execPath;

    public FilesetSymlinkFile(Artifact fileset, PathFragment execPath) {
      this.fileset = fileset;
      this.execPath = execPath;
    }

    private PathFragment getExecPath() {
      return execPath;
    }

    @Override
    public String getDirname() {
      PathFragment parent = getExecPath().getParentDirectory();
      return (parent == null) ? "/" : parent.getSafePathString();
    }

    @Override
    public String getFilename() {
      return getExecPath().getBaseName();
    }

    @Override
    public String getExtension() {
      return getExecPath().getFileExtension();
    }

    @Override
    public Label getOwnerLabel() {
      return fileset.getOwnerLabel();
    }

    @Override
    public FileRootApi getRoot() {
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
    public String getExecPathString() {
      return getExecPath().getPathString();
    }

    @Override
    public String expandToCommandLine() {
      return getExecPathString();
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
