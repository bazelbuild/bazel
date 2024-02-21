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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.util.Set;
import javax.annotation.Nullable;

/** Used by {@link ParallelEvaluator} to produce and consume {@link ErrorInfo} instances. */
public interface ErrorInfoManager {
  ErrorInfo fromException(
      SkyKey key,
      ReifiedSkyFunctionException skyFunctionException,
      boolean isTransitivelyTransient);

  /**
   * Returns the {@link ErrorInfo} to use when there isn't currently one because {@link
   * SkyFunction#compute} didn't throw a {@link SkyFunctionException}.
   */
  @Nullable
  ErrorInfo getErrorInfoToUse(SkyKey skyKey, boolean hasValue, Set<ErrorInfo> childErrorInfos);

  /**
   * Trivial {@link ErrorInfoManager} implementation whose {@link #fromException} simply uses
   * {@link ErrorInfo#fromException} and whose {@link #getErrorInfoToUse} makes an {@link ErrorInfo}
   * from the given {@code childErrorInfos}.
   */
  static class UseChildErrorInfoIfNecessary implements ErrorInfoManager {
    public static final UseChildErrorInfoIfNecessary INSTANCE = new UseChildErrorInfoIfNecessary();

    private UseChildErrorInfoIfNecessary() {
    }

    @Override
    public ErrorInfo fromException(
        SkyKey key,
        ReifiedSkyFunctionException skyFunctionException,
        boolean isTransitivelyTransient) {
      return ErrorInfo.fromException(skyFunctionException, isTransitivelyTransient);
    }

    @Override
    @Nullable
    public ErrorInfo getErrorInfoToUse(
        SkyKey skyKey, boolean hasValue, Set<ErrorInfo> childErrorInfos) {
      if (childErrorInfos.isEmpty()) {
        return null;
      }
      var errorInfo = ErrorInfo.fromChildErrors(skyKey, childErrorInfos);

      return hasValue ? ErrorInfo.withValue(errorInfo) : errorInfo;
    }
  }
}
