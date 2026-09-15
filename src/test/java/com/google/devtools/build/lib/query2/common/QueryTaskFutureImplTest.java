// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assert.assertThrows;

import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment.QueryTaskFutureImpl;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery.Code;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class QueryTaskFutureImplTest {
  @Test
  public void whenSucceedsOrIsCancelledCall_inputFutureSuccess()
      throws InterruptedException, ExecutionException {
    SettableFuture<Void> inputFuture = SettableFuture.create();

    QueryTaskFuture<String> nextFuture =
        QueryTaskFutureImpl.whenSucceedsOrIsCancelledCall(
            QueryTaskFutureImpl.ofDelegate(inputFuture),
            () -> "Callback Return",
            MoreExecutors.listeningDecorator(Executors.newSingleThreadExecutor()));

    inputFuture.setFuture(immediateVoidFuture());
    var unused = ((QueryTaskFutureImpl<String>) nextFuture).get();
    assertThat(nextFuture.getIfSuccessful()).isEqualTo("Callback Return");
  }

  @Test
  public void whenSucceedsOrCancelsCall_inputFutureCancels()
      throws InterruptedException, ExecutionException {
    QueryTaskFutureImpl<Void> inputQueryTaskFutureImpl =
        QueryTaskFutureImpl.ofDelegate(SettableFuture.create());

    QueryTaskFuture<String> nextFuture =
        QueryTaskFutureImpl.whenSucceedsOrIsCancelledCall(
            inputQueryTaskFutureImpl,
            () -> "Callback Return",
            MoreExecutors.listeningDecorator(Executors.newSingleThreadExecutor()));

    var unused1 = inputQueryTaskFutureImpl.cancel(true);
    var unused2 = ((QueryTaskFutureImpl<String>) nextFuture).get();
    assertThat(nextFuture.getIfSuccessful()).isEqualTo("Callback Return");
  }

  @Test
  public void whenSucceedsOrCancelsCall_inputFutureFails()
      throws InterruptedException, ExecutionException {
    SettableFuture<Void> inputFuture = SettableFuture.create();

    QueryTaskFuture<String> nextFuture =
        QueryTaskFutureImpl.whenSucceedsOrIsCancelledCall(
            QueryTaskFutureImpl.ofDelegate(inputFuture),
            () -> "Callback Return",
            MoreExecutors.listeningDecorator(Executors.newSingleThreadExecutor()));

    QueryException queryException =
        new QueryException("Deliberate failure", Code.ACTION_QUERY_UNKNOWN);
    inputFuture.setException(queryException);

    ExecutionException thrownFromDirectGet =
        assertThrows(ExecutionException.class, ((QueryTaskFutureImpl<String>) nextFuture)::get);
    Throwable cause = thrownFromDirectGet.getCause();
    assertThat(cause).isInstanceOf(QueryException.class);
    assertThat(thrownFromDirectGet).hasMessageThat().contains("Deliberate failure");

    IllegalStateException thrownFromGetIfSuccessful =
        assertThrows(IllegalStateException.class, nextFuture::getIfSuccessful);
    assertThat(thrownFromGetIfSuccessful)
        .hasCauseThat()
        .hasMessageThat()
        .isEqualTo(thrownFromDirectGet.getMessage());
  }
}
