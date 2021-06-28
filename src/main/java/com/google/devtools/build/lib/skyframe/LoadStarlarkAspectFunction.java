// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Objects;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction to load aspects from Starlark extensions and return StarlarkAspect.
 *
 * <p>Used for loading top-level aspects. At top level, in {@link
 * com.google.devtools.build.lib.analysis.BuildView}, we cannot invoke two SkyFunctions one after
 * another, so BuildView calls this function to do the work.
 */
public class LoadStarlarkAspectFunction implements SkyFunction {
  private static final Interner<StarlarkAspectLoadingKey> starlarkAspectLoadingKeyInterner =
      BlazeInterners.newWeakInterner();

  LoadStarlarkAspectFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws LoadStarlarkAspectFunctionException, InterruptedException {
    StarlarkAspectLoadingKey aspectLoadingKey = (StarlarkAspectLoadingKey) skyKey.argument();

    Label extensionLabel = aspectLoadingKey.getAspectClass().getExtensionLabel();
    String exportedName = aspectLoadingKey.getAspectClass().getExportedName();
    StarlarkAspect starlarkAspect;
    try {
      starlarkAspect = AspectFunction.loadStarlarkAspect(env, extensionLabel, exportedName);
      if (starlarkAspect == null) {
        return null;
      }
      if (!starlarkAspect.getParamAttributes().isEmpty()) {
        String msg =
            String.format(
                "Cannot instantiate parameterized aspect %s at the top level.",
                starlarkAspect.getName());
        throw new AspectCreationException(
            msg,
            new LabelCause(
                extensionLabel,
                createDetailedCode(msg, Code.PARAMETERIZED_TOP_LEVEL_ASPECT_INVALID)));
      }
    } catch (AspectCreationException e) {
      throw new LoadStarlarkAspectFunctionException(e);
    }

    return new StarlarkAspectLoadingValue(starlarkAspect);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static DetailedExitCode createDetailedCode(String msg, Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(msg)
            .setAnalysis(Analysis.newBuilder().setCode(code))
            .build());
  }

  /** Exceptions thrown from LoadStarlarkAspectFunction. */
  public static class LoadStarlarkAspectFunctionException extends SkyFunctionException {
    public LoadStarlarkAspectFunctionException(AspectCreationException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }

  public static StarlarkAspectLoadingKey createStarlarkAspectLoadingKey(
      StarlarkAspectClass aspectClass) {
    return StarlarkAspectLoadingKey.createInternal(aspectClass);
  }

  /** Skykey for loading Starlark aspect. */
  @AutoCodec
  public static final class StarlarkAspectLoadingKey implements SkyKey {
    private final StarlarkAspectClass aspectClass;
    private final int hashCode;

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static StarlarkAspectLoadingKey createInternal(StarlarkAspectClass aspectClass) {
      return starlarkAspectLoadingKeyInterner.intern(
          new StarlarkAspectLoadingKey(aspectClass, java.util.Objects.hashCode(aspectClass)));
    }

    private StarlarkAspectLoadingKey(StarlarkAspectClass aspectClass, int hashCode) {
      this.aspectClass = aspectClass;
      this.hashCode = hashCode;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.LOAD_STARLARK_ASPECT;
    }

    StarlarkAspectClass getAspectClass() {
      return aspectClass;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof StarlarkAspectLoadingKey)) {
        return false;
      }
      StarlarkAspectLoadingKey that = (StarlarkAspectLoadingKey) o;
      return hashCode == that.hashCode && Objects.equal(aspectClass, that.aspectClass);
    }

    @Override
    public String toString() {
      return aspectClass.toString();
    }
  }

  /** SkyValue for {@code StarlarkAspectLoadingKey} holds the loaded {@code StarlarkAspect}. */
  public static class StarlarkAspectLoadingValue implements SkyValue {
    private final StarlarkAspect starlarkAspect;

    public StarlarkAspectLoadingValue(StarlarkAspect starlarkAspect) {
      this.starlarkAspect = starlarkAspect;
    }

    public StarlarkAspect getAspect() {
      return starlarkAspect;
    }
  }
}
