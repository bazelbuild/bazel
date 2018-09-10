// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationSuccessState;

/**
 * Supplier of {@link EvaluationSuccessState} that crashes if its contained {@link NodeEntry} throws
 * an {@link InterruptedException} on value retrieval.
 */
class EvaluationSuccessStateSupplier implements Supplier<EvaluationSuccessState> {
  private final NodeEntry state;

  EvaluationSuccessStateSupplier(NodeEntry state) {
    this.state = state;
  }

  @Override
  public EvaluationSuccessState get() {
    try {
      return state.getValue() != null
          ? EvaluationSuccessState.SUCCESS
          : EvaluationSuccessState.FAILURE;
    } catch (InterruptedException e) {
      throw new IllegalStateException(
          "Graph implementations in which value retrieval can block should not be used in "
              + "frameworks that use the value in EvaluationProgressReceiver, since that could "
              + "result in significant slowdowns: "
              + state,
          e);
    }
  }

  static Supplier<EvaluationSuccessState> fromSkyValue(SkyValue value) {
    return ValueWithMetadata.justValue(value) != null
        ? Suppliers.ofInstance(EvaluationSuccessState.SUCCESS)
        : Suppliers.ofInstance(EvaluationSuccessState.FAILURE);
  }
}
