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
package com.google.devtools.build.skyframe;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** An environment that can observe the deps requested through getValue(s) calls. */
public class RecordingSkyFunctionEnvironment implements Environment {

  private final Environment delegate;
  private final Consumer<SkyKey> skyKeyReceiver;
  private final Consumer<Iterable<SkyKey>> skyKeysReceiver;
  private final Consumer<Exception> exceptionReceiver;

  public RecordingSkyFunctionEnvironment(
      Environment delegate,
      Consumer<SkyKey> skyKeyReceiver,
      Consumer<Iterable<SkyKey>> skyKeysReceiver,
      Consumer<Exception> exceptionReceiver) {
    this.delegate = delegate;
    this.skyKeyReceiver = skyKeyReceiver;
    this.skyKeysReceiver = skyKeysReceiver;
    this.exceptionReceiver = exceptionReceiver;
  }

  private void recordDep(SkyKey key) {
    skyKeyReceiver.accept(key);
  }

  @SuppressWarnings("unchecked") // Cast Iterable<? extends SkyKey> to Iterable<SkyKey>.
  private void recordDeps(Iterable<? extends SkyKey> keys) {
    skyKeysReceiver.accept((Iterable<SkyKey>) keys);
  }

  private void noteException(Exception e) {
    exceptionReceiver.accept(e);
  }

  public Environment getDelegate() {
    return delegate;
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey valueName) throws InterruptedException {
    recordDep(valueName);
    return delegate.getValue(valueName);
  }

  @Nullable
  @Override
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E, InterruptedException {
    recordDep(depKey);
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    recordDep(depKey);
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    recordDep(depKey);
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2, exceptionClass3);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    recordDep(depKey);
    try {
      return delegate.getValueOrThrow(
          depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Nullable
  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
          throws E1, E2, E3, E4, E5, InterruptedException {
    recordDep(depKey);
    try {
      return delegate.getValueOrThrow(
          depKey,
          exceptionClass1,
          exceptionClass2,
          exceptionClass3,
          exceptionClass4,
          exceptionClass5);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public Map<SkyKey, SkyValue> getValues(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    recordDeps(depKeys);
    return delegate.getValues(depKeys);
  }

  @Override
  public <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
      Iterable<? extends SkyKey> depKeys, Class<E> exceptionClass) throws InterruptedException {
    recordDeps(depKeys);
    try {
      return delegate.getValuesOrThrow(depKeys, exceptionClass);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public <E1 extends Exception, E2 extends Exception>
      Map<SkyKey, ValueOrException2<E1, E2>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
          throws InterruptedException {
    recordDeps(depKeys);
    try {
      return delegate.getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws InterruptedException {
    recordDeps(depKeys);
    try {
      return delegate.getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2, exceptionClass3);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      Map<SkyKey, ValueOrException4<E1, E2, E3, E4>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws InterruptedException {
    recordDeps(depKeys);
    try {
      return delegate.getValuesOrThrow(
          depKeys, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
          throws InterruptedException {
    recordDeps(depKeys);
    try {
      return delegate.getValuesOrThrow(
          depKeys,
          exceptionClass1,
          exceptionClass2,
          exceptionClass3,
          exceptionClass4,
          exceptionClass5);
    } catch (Exception e) {
      noteException(e);
      throw e;
    }
  }

  @Override
  public boolean valuesMissing() {
    return delegate.valuesMissing();
  }

  @Override
  public ExtendedEventHandler getListener() {
    return delegate.getListener();
  }

  @Override
  public boolean inErrorBubblingForTesting() {
    return delegate.inErrorBubblingForTesting();
  }

  @Nullable
  @Override
  public GroupedList<SkyKey> getTemporaryDirectDeps() {
    return delegate.getTemporaryDirectDeps();
  }

  @Override
  public void injectVersionForNonHermeticFunction(Version version) {
    delegate.injectVersionForNonHermeticFunction(version);
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    delegate.dependOnFuture(future);
  }

  @Override
  public boolean restartPermitted() {
    return delegate.restartPermitted();
  }
}
