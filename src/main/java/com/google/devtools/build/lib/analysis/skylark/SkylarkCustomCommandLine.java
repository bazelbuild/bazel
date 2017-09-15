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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
import java.util.ArrayList;
import java.util.IllegalFormatException;
import java.util.List;
import javax.annotation.Nullable;

/** Supports ctx.actions.args() from Skylark. */
class SkylarkCustomCommandLine extends CommandLine {
  private final SkylarkSemanticsOptions skylarkSemantics;
  private final EventHandler eventHandler;
  private final ImmutableList<Object> arguments;

  private static final Joiner LINE_JOINER = Joiner.on("\n").skipNulls();
  private static final Joiner FIELD_JOINER = Joiner.on(": ").skipNulls();

  static final class VectorArg {
    private static Interner<VectorArg> interner = BlazeInterners.newStrongInterner();

    private final boolean isNestedSet;
    private final boolean hasFormat;
    private final boolean hasBeforeEach;
    private final boolean hasJoinWith;
    private final boolean hasMapFn;
    private final boolean hasLocation;

    VectorArg(
        boolean isNestedSet,
        boolean hasFormat,
        boolean hasBeforeEach,
        boolean hasJoinWith,
        boolean hasMapFn,
        boolean hasLocation) {
      this.isNestedSet = isNestedSet;
      this.hasFormat = hasFormat;
      this.hasBeforeEach = hasBeforeEach;
      this.hasJoinWith = hasJoinWith;
      this.hasMapFn = hasMapFn;
      this.hasLocation = hasLocation;
    }

    private static void push(ImmutableList.Builder<Object> arguments, Builder arg) {
      boolean wantsLocation = arg.format != null || arg.mapFn != null;
      boolean hasLocation = arg.location != null && wantsLocation;
      VectorArg vectorArg =
          new VectorArg(
              arg.nestedSet != null,
              arg.format != null,
              arg.beforeEach != null,
              arg.joinWith != null,
              arg.mapFn != null,
              hasLocation);
      vectorArg = interner.intern(vectorArg);
      arguments.add(vectorArg);
      if (vectorArg.isNestedSet) {
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
      if (vectorArg.hasMapFn) {
        arguments.add(arg.mapFn);
      }
      if (vectorArg.hasFormat) {
        arguments.add(arg.format);
      }
      if (vectorArg.hasBeforeEach) {
        arguments.add(arg.beforeEach);
      }
      if (vectorArg.hasJoinWith) {
        arguments.add(arg.joinWith);
      }
    }

    private int eval(
        List<Object> arguments,
        int argi,
        ImmutableList.Builder<String> builder,
        SkylarkSemanticsOptions skylarkSemantics,
        EventHandler eventHandler)
        throws CommandLineExpansionException {
      final List<Object> mutatedValues;
      final int count;
      if (isNestedSet) {
        NestedSet<?> nestedSet = (NestedSet<?>) arguments.get(argi++);
        mutatedValues = Lists.newArrayList(nestedSet);
        count = mutatedValues.size();
      } else {
        count = (Integer) arguments.get(argi++);
        mutatedValues = new ArrayList<>(count);
        for (int i = 0; i < count; ++i) {
          mutatedValues.add(arguments.get(argi++));
        }
      }
      final Location location = hasLocation ? (Location) arguments.get(argi++) : null;
      if (hasMapFn) {
        BaseFunction mapFn = (BaseFunction) arguments.get(argi++);
        Object result = applyMapFn(mapFn, mutatedValues, location, skylarkSemantics, eventHandler);
        if (!(result instanceof List)) {
          throw new CommandLineExpansionException(
              errorMessage(
                  "map_fn must return a list, got " + result.getClass().getSimpleName(),
                  location,
                  null));
        }
        List resultAsList = (List) result;
        mutatedValues.clear();
        mutatedValues.addAll(resultAsList);
      }
      for (int i = 0; i < count; ++i) {
        mutatedValues.set(i, valueToString(mutatedValues.get(i)));
      }
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        Formatter formatter = new Formatter(formatStr, location);
        try {
          for (int i = 0; i < count; ++i) {
            mutatedValues.set(i, formatter.format(mutatedValues.get(i)));
          }
        } catch (IllegalFormatException e) {
          throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, null));
        }
      }
      if (hasBeforeEach) {
        String beforeEach = (String) arguments.get(argi++);
        for (int i = 0; i < count; ++i) {
          builder.add(beforeEach);
          builder.add((String) mutatedValues.get(i));
        }
      } else if (hasJoinWith) {
        String joinWith = (String) arguments.get(argi++);
        builder.add(Joiner.on(joinWith).join(mutatedValues));
      } else {
        for (int i = 0; i < count; ++i) {
          builder.add((String) mutatedValues.get(i));
        }
      }
      return argi;
    }

    static class Builder {
      @Nullable private final SkylarkList<?> list;
      @Nullable private final NestedSet<?> nestedSet;
      private String format;
      private String beforeEach;
      private String joinWith;
      private Location location;
      private BaseFunction mapFn;

      public Builder(SkylarkList<?> list) {
        this.list = list;
        this.nestedSet = null;
      }

      public Builder(NestedSet<?> nestedSet) {
        this.list = null;
        this.nestedSet = nestedSet;
      }

      Builder setLocation(Location location) {
        this.location = location;
        return this;
      }

      Builder setFormat(String format) {
        this.format = format;
        return this;
      }

      Builder setBeforeEach(String beforeEach) {
        this.beforeEach = beforeEach;
        return this;
      }

      public Builder setJoinWith(String joinWith) {
        this.joinWith = joinWith;
        return this;
      }

      public Builder setMapFn(BaseFunction mapFn) {
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
      VectorArg vectorArg = (VectorArg) o;
      return isNestedSet == vectorArg.isNestedSet
          && hasFormat == vectorArg.hasFormat
          && hasBeforeEach == vectorArg.hasBeforeEach
          && hasJoinWith == vectorArg.hasJoinWith
          && hasMapFn == vectorArg.hasMapFn
          && hasLocation == vectorArg.hasLocation;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(
          isNestedSet, hasFormat, hasBeforeEach, hasJoinWith, hasMapFn, hasLocation);
    }
  }

  static final class ScalarArg {
    private static Interner<ScalarArg> interner = BlazeInterners.newStrongInterner();

    private final boolean hasFormat;
    private final boolean hasMapFn;
    private final boolean hasLocation;

    public ScalarArg(boolean hasFormat, boolean hasMapFn, boolean hasLocation) {
      this.hasFormat = hasFormat;
      this.hasMapFn = hasMapFn;
      this.hasLocation = hasLocation;
    }

    private static void push(ImmutableList.Builder<Object> arguments, Builder arg) {
      boolean wantsLocation = arg.format != null || arg.mapFn != null;
      boolean hasLocation = arg.location != null && wantsLocation;
      ScalarArg scalarArg = new ScalarArg(arg.format != null, arg.mapFn != null, hasLocation);
      scalarArg = interner.intern(scalarArg);
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
        SkylarkSemanticsOptions skylarkSemantics,
        EventHandler eventHandler)
        throws CommandLineExpansionException {
      Object object = arguments.get(argi++);
      final Location location = hasLocation ? (Location) arguments.get(argi++) : null;
      if (hasMapFn) {
        BaseFunction mapFn = (BaseFunction) arguments.get(argi++);
        object = applyMapFn(mapFn, object, location, skylarkSemantics, eventHandler);
      }
      object = valueToString(object);
      if (hasFormat) {
        String formatStr = (String) arguments.get(argi++);
        Formatter formatter = new Formatter(formatStr, location);
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

      public Builder setMapFn(BaseFunction mapFn) {
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
    private final SkylarkSemanticsOptions skylarkSemantics;
    private final ImmutableList.Builder<Object> arguments = ImmutableList.builder();
    private final EventHandler eventHandler;

    public Builder(SkylarkSemanticsOptions skylarkSemantics, EventHandler eventHandler) {
      this.skylarkSemantics = skylarkSemantics;
      this.eventHandler = eventHandler;
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
      return new SkylarkCustomCommandLine(this);
    }
  }

  SkylarkCustomCommandLine(Builder builder) {
    this.arguments = builder.arguments.build();
    this.skylarkSemantics = builder.skylarkSemantics;
    this.eventHandler = builder.eventHandler;
  }

  @Override
  public Iterable<String> arguments() throws CommandLineExpansionException {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (int argi = 0; argi < arguments.size(); ) {
      Object arg = arguments.get(argi++);
      if (arg instanceof VectorArg) {
        argi = ((VectorArg) arg).eval(arguments, argi, result, skylarkSemantics, eventHandler);
      } else if (arg instanceof ScalarArg) {
        argi = ((ScalarArg) arg).eval(arguments, argi, result, skylarkSemantics, eventHandler);
      } else {
        result.add(valueToString(arg));
      }
    }
    return result.build();
  }

  private static String valueToString(Object value) {
    if (value instanceof Artifact) {
      Artifact artifact = (Artifact) value;
      return artifact.getExecPath().getPathString();
    }
    return value.toString();
  }

  private static class Formatter {
    private final String formatStr;
    @Nullable private final Location location;
    private final ArrayList<Object> args;

    public Formatter(String formatStr, Location location) {
      this.formatStr = formatStr;
      this.location = location;
      this.args = new ArrayList<>(1); // Reused arg list to reduce GC
      this.args.add(null);
    }

    String format(Object object) throws CommandLineExpansionException {
      try {
        args.set(0, object);
        return Printer.getPrinter().formatWithList(formatStr, args).toString();
      } catch (IllegalFormatException e) {
        throw new CommandLineExpansionException(errorMessage(e.getMessage(), location, null));
      }
    }
  }

  private static Object applyMapFn(
      BaseFunction mapFn,
      Object arg,
      Location location,
      SkylarkSemanticsOptions skylarkSemantics,
      EventHandler eventHandler)
      throws CommandLineExpansionException {
    ImmutableList<Object> args = ImmutableList.of(arg);
    try (Mutability mutability = Mutability.create("map_fn")) {
      Environment env =
          Environment.builder(mutability)
              .setSemantics(skylarkSemantics)
              .setEventHandler(eventHandler)
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
