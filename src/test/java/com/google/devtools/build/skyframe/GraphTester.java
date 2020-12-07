// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A helper class to create graphs and run skyframe tests over these graphs.
 *
 * <p>There are two types of values, computing values, which may not be set to a constant value,
 * and leaf values, which must be set to a constant value and may not have any dependencies.
 *
 * <p>Note that the value builder looks into the test values created here to determine how to
 * behave. However, skyframe will only re-evaluate the value and call the value builder if any of
 * its dependencies has changed. That means in order to change the set of dependencies of a value,
 * you need to also change one of its previous dependencies to force re-evaluation. Changing a
 * computing value does not mark it as modified.
 */
public class GraphTester {

  public static final SkyFunctionName NODE_TYPE = SkyFunctionName.FOR_TESTING;
  private final Map<SkyFunctionName, SkyFunction> functionMap = new HashMap<>();

  private final Map<SkyKey, TestFunction> values = new HashMap<>();
  private final Set<SkyKey> modifiedValues = new LinkedHashSet<>();

  public GraphTester() {
    functionMap.put(NODE_TYPE, new DelegatingFunction());
    functionMap.put(FOR_TESTING_NONHERMETIC, new DelegatingFunction());
  }

  public TestFunction getOrCreate(String name) {
    return getOrCreate(skyKey(name));
  }

  public TestFunction getOrCreate(SkyKey key) {
    return getOrCreate(key, false);
  }

  public TestFunction getOrCreate(SkyKey key, boolean markAsModified) {
    TestFunction result = values.get(key);
    if (result == null) {
      result = new TestFunction();
      values.put(key, result);
    } else if (markAsModified) {
      modifiedValues.add(key);
    }
    return result;
  }

  public TestFunction set(String key, SkyValue value) {
    return set(skyKey(key), value);
  }

  public TestFunction set(SkyKey key, SkyValue value) {
    return getOrCreate(key, true).setConstantValue(value);
  }

  public ImmutableSet<SkyKey> getModifiedValues() {
    return ImmutableSet.copyOf(modifiedValues);
  }

  public void clearModifiedValues() {
    modifiedValues.clear();
  }

  public SkyFunction getFunction() {
    return new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey key, Environment env)
          throws SkyFunctionException, InterruptedException {
        TestFunction builder = values.get(key);
        Preconditions.checkState(builder != null, "No TestFunction for " + key);
        if (builder.builder != null) {
          return builder.builder.compute(key, env);
        }
        if (builder.warning != null) {
          env.getListener().handle(Event.warn(builder.warning));
        }
        if (builder.progress != null) {
          env.getListener().handle(Event.progress(builder.progress));
        }
        if (builder.postable != null) {
          env.getListener().post(builder.postable);
        }
        Map<SkyKey, SkyValue> deps = new LinkedHashMap<>();
        boolean oneMissing = false;
        for (Pair<SkyKey, SkyValue> dep : builder.deps) {
          SkyValue value;
          if (dep.second == null) {
            value = env.getValue(dep.first);
          } else {
            try {
              value = env.getValueOrThrow(dep.first, SomeErrorException.class);
            } catch (SomeErrorException e) {
              value = dep.second;
            }
          }
          if (value == null) {
            oneMissing = true;
          } else {
            deps.put(dep.first, value);
          }
          Preconditions.checkState(
              oneMissing == env.valuesMissing(), "%s %s %s", dep, value, env.valuesMissing());
        }
        if (env.valuesMissing()) {
          return null;
        }

        if (builder.hasTransientError) {
          throw new GenericFunctionException(new SomeErrorException(key.toString()),
              Transience.TRANSIENT);
        }
        if (builder.hasError) {
          throw new GenericFunctionException(new SomeErrorException(key.toString()),
              Transience.PERSISTENT);
        }

        if (builder.value != null) {
          return builder.value;
        }

        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException(key.toString());
        }

        return builder.computer.compute(deps, env);
      }

      @Nullable
      @Override
      public String extractTag(SkyKey skyKey) {
        return values.get(skyKey).tag;
      }
    };
  }

  public static SkyKey skyKey(String key) {
    return Key.create(key);
  }

  public static NonHermeticKey nonHermeticKey(String key) {
    return NonHermeticKey.create(key);
  }

  /** A value in the testing graph that is constructed in the tester. */
  public static class TestFunction {
    // TODO(bazel-team): We could use a multiset here to simulate multi-pass dependency discovery.
    private final Set<Pair<SkyKey, SkyValue>> deps = new LinkedHashSet<>();
    private SkyValue value;
    private ValueComputer computer;
    private SkyFunction builder = null;

    private boolean hasTransientError;
    private boolean hasError;

    private String warning;
    private String progress;
    private Postable postable;

    private String tag;

    public TestFunction addDependency(String name) {
      return addDependency(skyKey(name));
    }

    public TestFunction addDependency(SkyKey key) {
      deps.add(Pair.<SkyKey, SkyValue>of(key, null));
      return this;
    }

    public TestFunction removeDependency(String name) {
      return removeDependency(skyKey(name));
    }

    public TestFunction removeDependency(SkyKey key) {
      deps.remove(Pair.<SkyKey, SkyValue>of(key, null));
      return this;
    }

    public TestFunction addErrorDependency(String name, SkyValue altValue) {
      return addErrorDependency(skyKey(name), altValue);
    }

    public TestFunction addErrorDependency(SkyKey key, SkyValue altValue) {
      deps.add(Pair.of(key, altValue));
      return this;
    }

    public TestFunction setConstantValue(SkyValue value) {
      Preconditions.checkState(this.computer == null);
      this.value = value;
      return this;
    }

    public TestFunction unsetConstantValue() {
      this.value = null;
      return this;
    }

    public TestFunction setComputedValue(ValueComputer computer) {
      Preconditions.checkState(this.value == null);
      this.computer = computer;
      return this;
    }

    public TestFunction unsetComputedValue() {
      this.computer = null;
      return this;
    }

    public TestFunction setBuilder(SkyFunction builder) {
      Preconditions.checkState(this.value == null);
      Preconditions.checkState(this.computer == null);
      Preconditions.checkState(deps.isEmpty());
      Preconditions.checkState(!hasTransientError);
      Preconditions.checkState(!hasError);
      Preconditions.checkState(warning == null);
      Preconditions.checkState(progress == null);
      this.builder = builder;
      return this;
    }

    public TestFunction setBuilderUnconditionally(SkyFunction builder) {
      this.builder = builder;
      return this;
    }

    public TestFunction setHasTransientError(boolean hasError) {
      this.hasTransientError = hasError;
      return this;
    }

    public TestFunction setHasError(boolean hasError) {
      // TODO(bazel-team): switch to an enum for hasError.
      this.hasError = hasError;
      return this;
    }

    public TestFunction setWarning(String warning) {
      this.warning = warning;
      return this;
    }

    public TestFunction setProgress(String info) {
      this.progress = info;
      return this;
    }

    public TestFunction setTag(String tag) {
      this.tag = tag;
      return this;
    }

    public TestFunction setPostable(Postable postable) {
      this.postable = postable;
      return this;
    }
  }

  public static ImmutableList<SkyKey> toSkyKeys(String... names) {
    ImmutableList.Builder<SkyKey> result = ImmutableList.builder();
    for (String element : names) {
      result.add(Key.create(element));
    }
    return result.build();
  }

  public static SkyKey toSkyKey(String name) {
    return toSkyKeys(name).get(0);
  }

  private class DelegatingFunction implements SkyFunction {
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
        InterruptedException {
      return getFunction().compute(skyKey, env);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return getFunction().extractTag(skyKey);
    }
  }

  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctionMap() {
    return ImmutableMap.copyOf(functionMap);
  }

  public void putSkyFunction(SkyFunctionName functionName, SkyFunction function) {
    functionMap.put(functionName, function);
  }

  /**
   * Simple value class that stores strings.
   */
  public static class StringValue implements SkyValue {
    protected final String value;

    public StringValue(String value) {
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof StringValue)) {
        return false;
      }
      return value.equals(((StringValue) o).value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }

    @Override
    public String toString() {
      return "StringValue: " + getValue();
    }

    public static StringValue of(String string) {
      return new StringValue(string);
    }

    public static StringValue from(SkyValue skyValue) {
      assertThat(skyValue).isInstanceOf(StringValue.class);
      return (StringValue) skyValue;
    }
  }

  /** A StringValue that is also a NotComparableSkyValue. */
  public static class NotComparableStringValue extends StringValue
          implements NotComparableSkyValue {
    public NotComparableStringValue(String value) {
      super(value);
    }

    @Override
    public boolean equals(Object o) {
      throw new UnsupportedOperationException(value + " is incomparable - what are you doing?");
    }

    @Override
    public int hashCode() {
      throw new UnsupportedOperationException(value + " is incomparable - what are you doing?");
    }
  }

  /** An unshareable version of {@link StringValue}. */
  public static final class UnshareableStringValue extends StringValue {
    public UnshareableStringValue(String value) {
      super(value);
    }

    @Override
    public boolean dataIsShareable() {
      return false;
    }
  }

  /**
   * A callback interface to provide the value computation.
   */
  public interface ValueComputer {
    /** This is called when all the declared dependencies exist. It may request new dependencies. */
    SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env)
        throws InterruptedException;
  }

  public static final ValueComputer COPY = new ValueComputer() {
    @Override
    public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env) {
      return Iterables.getOnlyElement(deps.values());
    }
  };

  public static final ValueComputer CONCATENATE = new ValueComputer() {
    @Override
    public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env) {
      StringBuilder result = new StringBuilder();
      for (SkyValue value : deps.values()) {
        result.append(((StringValue) value).value);
      }
      return new StringValue(result.toString());
    }
  };

  public static ValueComputer formatter(final SkyKey key, final String format) {
    return new ValueComputer() {
      @Override
      public SkyValue compute(Map<SkyKey, SkyValue> deps, Environment env)
          throws InterruptedException {
        return StringValue.of(String.format(format, StringValue.from(deps.get(key)).getValue()));
      }
    };
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<String> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(String arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(String arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class NonHermeticKey extends AbstractSkyKey<String> {
    private static final Interner<NonHermeticKey> interner = BlazeInterners.newWeakInterner();

    private NonHermeticKey(String arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static NonHermeticKey create(String arg) {
      return interner.intern(new NonHermeticKey(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return FOR_TESTING_NONHERMETIC;
    }
  }

  private static final SkyFunctionName FOR_TESTING_NONHERMETIC =
      SkyFunctionName.createNonHermetic("FOR_TESTING_NONHERMETIC");
}
