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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.build.skyframe.ValueOrException2;
import com.google.devtools.build.skyframe.ValueOrException3;
import com.google.devtools.build.skyframe.ValueOrException4;
import com.google.devtools.build.skyframe.ValueOrException5;
import com.google.devtools.build.skyframe.Version;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction.Environment} which returns a {@link
 * ExtendedEventHandler.ProgressLike}-suppressing {@link ExtendedEventHandler} from {@link
 * #getListener()}.
 *
 * <p>Otherwise, delegates calls to its wrapped {@link SkyFunction.Environment}.
 */
class ProgressEventSuppressingEnvironment implements SkyFunction.Environment {

  private final SkyFunction.Environment delegate;
  private final ProgressSuppressingEventHandler suppressingEventHandler;

  ProgressEventSuppressingEnvironment(SkyFunction.Environment env) {
    this.delegate = env;
    this.suppressingEventHandler = new ProgressSuppressingEventHandler(env.getListener());
  }

  @Override
  public ExtendedEventHandler getListener() {
    return suppressingEventHandler;
  }

  @Override
  @Nullable
  public SkyValue getValue(SkyKey valueName) throws InterruptedException {
    return delegate.getValue(valueName);
  }

  @Override
  @Nullable
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass);
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2);
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2, exceptionClass3);
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    return delegate.getValueOrThrow(
        depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
  }

  @Override
  @Nullable
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
    return delegate.getValueOrThrow(
        depKey,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5);
  }

  @Override
  public Map<SkyKey, SkyValue> getValues(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    return delegate.getValues(depKeys);
  }

  @Override
  public <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
      Iterable<? extends SkyKey> depKeys, Class<E> exceptionClass) throws InterruptedException {
    return delegate.getValuesOrThrow(depKeys, exceptionClass);
  }

  @Override
  public <E1 extends Exception, E2 extends Exception>
      Map<SkyKey, ValueOrException2<E1, E2>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
          throws InterruptedException {
    return delegate.getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2);
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws InterruptedException {
    return delegate.getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2, exceptionClass3);
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
    return delegate.getValuesOrThrow(
        depKeys, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
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
    return delegate.getValuesOrThrow(
        depKeys,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5);
  }

  @Override
  public boolean valuesMissing() {
    return delegate.valuesMissing();
  }

  @Override
  @Nullable
  public GroupedList<SkyKey> getTemporaryDirectDeps() {
    return delegate.getTemporaryDirectDeps();
  }

  @Override
  public void injectVersionForNonHermeticFunction(Version version) {
    delegate.injectVersionForNonHermeticFunction(version);
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) throws InterruptedException {
    delegate.registerDependencies(keys);
  }

  @Override
  @VisibleForTesting
  public boolean inErrorBubblingForTesting() {
    return delegate.inErrorBubblingForTesting();
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
