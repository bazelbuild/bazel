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
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IllegalFormatException;
import java.util.List;
import javax.annotation.Nullable;

/** Supports ctx.actions.args() from Skylark. */
@AutoCodec
class SkylarkCustomCommandLine extends CommandLine {
  private final SkylarkSemantics skylarkSemantics;
  private final ImmutableList<Object> arguments;

  private static final Joiner LINE_JOINER = Joiner.on("\n").skipNulls();
  private static final Joiner FIELD_JOINER = Joiner.on(": ").skipNulls();

  @AutoCodec
  static final class VectorArg {
    private static Interner<VectorArg> interner = BlazeInterners.newStrongInterner();

    private static final int IS_NESTED_SET = 1;
    private static final int HAS_LOCATION = 1 << 1;
    private static final int HAS_ARG_NAME = 1 << 2;
    private static final int HAS_MAP_ALL = 1 << 3;
    private static final int HAS_MAP_EACH = 1 << 4;
    private static final int HAS_FORMAT_EACH = 1 << 5;
    private static final int HAS_BEFORE_EACH = 1 << 6;
    private static final int HAS_JOIN_WITH = 1 << 7;
    private static final int HAS_FORMAT_JOINED = 1 << 8;
    private static final int OMIT_IF_EMPTY = 1 << 9;
    private static final int UNIQUIFY = 1 << 10;
    private static final int HAS_TERMINATE_WITH = 1 << 11;

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
      features |= arg.nestedSet != null ? IS_NESTED_SET : 0;
      features |= arg.argName != null ? HAS_ARG_NAME : 0;
      features |= arg.mapAll != null ? HAS_MAP_ALL : 0;
      features |= arg.mapEach != null ? HAS_MAP_EACH : 0;
      features |= arg.formatEach != null ? HAS_FORMAT_EACH : 0;
      features |= arg.beforeEach != null ? HAS_BEFORE_EACH : 0;
      features |= arg.joinWith != null ? HAS_JOIN_WITH : 0;
      features |= arg.formatJoined != null ? HAS_FORMAT_JOINED : 0;
      features |= arg.omitIfEmpty ? OMIT_IF_EMPTY : 0;
      features |= arg.uniquify ? UNIQUIFY : 0;
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
      if (arg.nestedSet != null) {
        arguments.add(arg.nestedSet);
      } else {
        ImmutableList<?> list = arg.list.getImmutableList();
        int count = list.size();
        arguments.add(count);
        for (int i = 0; i < count; ++i) {
          arguments.add(list.get(i));
        }
      }
      if (hasLocation) {
        arguments.add(arg.location);
      }
      if (arg.argName != null) {
        arguments.add(arg.argName);
      }
      if (arg.mapAll != null) {
        arguments.add(arg.mapAll);
      }
      if (arg.mapEach != null) {
        arguments.add(arg.mapEach);
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
        SkylarkSemantics skylarkSemantics)
        throws CommandLineExpansionException {
      final List<Object> originalValues;
      if ((features & IS_NESTED_SET) != 0) {
        NestedSet<Object> nestedSet = (NestedSet<Object>) arguments.get(argi++);
        originalValues = nestedSet.toList();
      } else {
        int count = (Integer) arguments.get(argi++);
        originalValues = arguments.subList(argi, argi + count);
        argi += count;
      }
      List<String> stringValues;
      final Location location =
          ((features & HAS_LOCATION) != 0) ? (Location) arguments.get(argi++) : null;
      final String argName =
          ((features & HAS_ARG_NAME) != 0) ? (String) arguments.get(argi++) : null;
      if ((features & HAS_MAP_EACH) != 0) {
        BaseFunction mapEach = (BaseFunction) arguments.get(argi++);
        stringValues = applyMapEach(mapEach, originalValues, location, skylarkSemantics);
      } else if ((features & HAS_MAP_ALL) != 0) {
        BaseFunction mapFn = (BaseFunction) arguments.get(argi++);
        Object result = applyMapFn(mapFn, originalValues, location, skylarkSemantics);
        if (!(result instanceof List)) {
          throw new CommandLineExpansionException(
              errorMessage(
                  "map_fn must return a list, got " + result.getClass().getSimpleName(),
                  location,
                  null));
        }
        List resultAsList = (List) result;
        if (resultAsList.size() != originalValues.size()) {
          throw new CommandLineExpansionException(
              errorMessage(
                  String.format(
                      "map_fn must return a list of the same length as the input. "
                          + "Found list of length %d, expected %d.",
                      resultAsList.size(), originalValues.size()),
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
        int count = originalValues.size();
        stringValues = new ArrayList<>(originalValues.size());
        for (int i = 0; i < count; ++i) {
          stringValues.add(CommandLineItem.expandToCommandLine(originalValues.get(i)));
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
      if (argName != null && !isEmptyAndShouldOmit) {
        builder.add(argName);
      }
      if ((features & HAS_FORMAT_EACH) != 0) {
        String formatStr = (String) arguments.get(argi++);
        Formatter formatter = new Formatter(formatStr, skylarkSemantics, location);
        try {
          int count = stringValues.size();
          for (int i = 0; i < count; ++i) {
            stringValues.set(i, formatter.format(stringValues.get(i)));
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
            Formatter formatter = new Formatter(formatJoined, skylarkSemantics, location);
            try {
              result = formatter.format(result);
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

    static class Builder {
      @Nullable private final SkylarkList<?> list;
      @Nullable private final NestedSet<?> nestedSet;
      private Location location;
      public String argName;
      private BaseFunction mapAll;
      private BaseFunction mapEach;
      private String formatEach;
      private String beforeEach;
      private String joinWith;
      private String formatJoined;
      private boolean omitIfEmpty;
      private boolean uniquify;
      private String terminateWith;

      Builder(SkylarkList<?> list) {
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
    private static Interner<ScalarArg> interner = BlazeInterners.newStrongInterner();

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
        SkylarkSemantics skylarkSemantics)
        throws CommandLineExpansionException {
      Object object = arguments.get(argi++);
      final Location location = hasLocation ? (Location) arguments.get(argi++) : null;
      if (hasMapFn) {
        BaseFunction mapFn = (BaseFunction) arguments.get(argi++);
        object = applyMapFn(mapFn, object, location, skylarkSemantics);
      }
      object = CommandLineItem.expandToCommandLine(object);
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        Formatter formatter = new Formatter(formatStr, skylarkSemantics, location);
        object = formatter.format(object);
      }
      builder.add((String) object);
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
    private final SkylarkSemantics skylarkSemantics;
    private final ImmutableList.Builder<Object> arguments = ImmutableList.builder();

    public Builder(SkylarkSemantics skylarkSemantics) {
      this.skylarkSemantics = skylarkSemantics;
    }

    void add(Object object) {
      arguments.add(object);
    }

    void add(VectorArg.Builder vectorArg) {
      VectorArg.push(arguments, vectorArg);
    }

    void add(ScalarArg.Builder scalarArg) {
      ScalarArg.push(arguments, scalarArg);
    }

    SkylarkCustomCommandLine build() {
      return new SkylarkCustomCommandLine(skylarkSemantics, arguments.build());
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  SkylarkCustomCommandLine(SkylarkSemantics skylarkSemantics, ImmutableList<Object> arguments) {
    this.arguments = arguments;
    this.skylarkSemantics = skylarkSemantics;
  }

  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (int argi = 0; argi < arguments.size(); ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi = ((VectorArg) arg).eval(arguments, argi, result, skylarkSemantics);
      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).eval(arguments, argi, result, skylarkSemantics);
      } else {
        result.add(CommandLineItem.expandToCommandLine(arg));
      }
    }
    return result.build();
  }

  private static class Formatter {
    private final String formatStr;
    private final SkylarkSemantics skylarkSemantics;
    @Nullable private final Location location;
    private final ArrayList<Object> args;

    public Formatter(String formatStr, SkylarkSemantics skylarkSemantics, Location location) {
      this.formatStr = formatStr;
      this.skylarkSemantics = skylarkSemantics;
      this.location = location;
      this.args = new ArrayList<>(1); // Reused arg list to reduce GC
      this.args.add(null);
    }

    String format(Object object) throws CommandLineExpansionException {
      try {
        args.set(0, object);
        SkylarkPrinter printer =
            skylarkSemantics.incompatibleDisallowOldStyleArgsAdd()
                ? Printer.getSimplifiedPrinter() : Printer.getPrinter();
        return printer.formatWithList(formatStr, args).toString();
      } catch (IllegalFormatException e) {
        throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, null));
      }
    }
  }

  private static Object applyMapFn(
      BaseFunction mapFn, Object arg, Location location, SkylarkSemantics skylarkSemantics)
      throws CommandLineExpansionException {
    ImmutableList<Object> args = ImmutableList.of(arg);
    try (Mutability mutability = Mutability.create("map_fn")) {
      Environment env =
          Environment.builder(mutability)
              .setSemantics(skylarkSemantics)
              .setEventHandler(NullEventHandler.INSTANCE)
              .build();
      return mapFn.call(args, ImmutableMap.of(), null, env);
    } catch (EvalException e) {
      throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, e.getCause()));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new CommandLineExpansionException(
          errorMessage("Thread was interrupted", location, null));
    }
  }

  private static ArrayList<String> applyMapEach(
      BaseFunction mapFn,
      List<Object> originalValues,
      Location location,
      SkylarkSemantics skylarkSemantics)
      throws CommandLineExpansionException {
    try (Mutability mutability = Mutability.create("map_each")) {
      Environment env =
          Environment.builder(mutability)
              .setSemantics(skylarkSemantics)
              // TODO(b/77140311): Error if we issue print statements
              .setEventHandler(NullEventHandler.INSTANCE)
              .build();
      Object[] args = new Object[1];
      int count = originalValues.size();
      ArrayList<String> stringValues = new ArrayList<>(count);
      for (int i = 0; i < count; ++i) {
        args[0] = originalValues.get(i);
        Object ret = mapFn.callWithArgArray(args, null, env, location);
        if (ret instanceof String) {
          stringValues.add((String) ret);
        } else if (ret instanceof SkylarkList) {
          for (Object val : ((SkylarkList) ret)) {
            if (!(val instanceof String)) {
              throw new CommandLineExpansionException(
                  "Expected map_each to return string, None, or list of strings, "
                      + "found list containing "
                      + val.getClass().getSimpleName());
            }
            stringValues.add((String) val);
          }
        } else if (ret != Runtime.NONE) {
          throw new CommandLineExpansionException(
              "Expected map_each to return string, None, or list of strings, found "
                  + ret.getClass().getSimpleName());
        }
      }
      return stringValues;
    } catch (EvalException e) {
      throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, e.getCause()));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new CommandLineExpansionException(
          errorMessage("Thread was interrupted", location, null));
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
}
