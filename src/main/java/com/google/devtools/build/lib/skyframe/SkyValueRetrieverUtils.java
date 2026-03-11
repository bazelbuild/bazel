// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.io.BaseEncoding.base16;

import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.DependOnFutureShim.DefaultDependOnFutureShim;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.CacheMissReason;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.NoCachedData;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.Restart;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalContext;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievedValue;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.SerializableSkyKeyComputeState;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheReaderDepsProvider;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisJsonLogWriter;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.time.Instant;
import java.util.function.Supplier;

/**
 * A wrapper around {@link SkyValueRetriever} to handle Bazel-on-Skyframe specific logic, metrics
 * gathering, and error handling.
 */
public final class SkyValueRetrieverUtils {

  public static RetrievalResult retrieveRemoteSkyValue(
      SkyKey key,
      Environment env,
      RemoteAnalysisCacheReaderDepsProvider analysisCachingDeps,
      Supplier<? extends SerializableSkyKeyComputeState> stateSupplier)
      throws InterruptedException {
    if (env.inErrorBubbling()) {
      // Remote retrieval during error bubbling causes incorrect error propagation. See b/449016469.
      return new NoCachedData(CacheMissReason.NOT_ATTEMPTED);
    }

    if (analysisCachingDeps.shouldBailOutOnMissingFingerprint()) {
      return new NoCachedData(CacheMissReason.NOT_ATTEMPTED);
    }

    Label label =
        switch (key) {
          case ActionLookupKey alk -> alk.getLabel();
          case ActionLookupData ald -> ald.getLabel();
          case Artifact artifact -> artifact.getOwnerLabel();
          default -> throw new IllegalStateException("unexpected key: " + key.getCanonicalName());
        };

    if (label == null) {
      // If there's no label, there's no cached data.
      return new NoCachedData(CacheMissReason.NOT_ATTEMPTED);
    }

    RetrievalResult retrievalResult = null;
    Exception exception = null;
    RetrievalContext state = env.getState(stateSupplier).getRetrievalContext();
    try {
      retrievalResult =
          SkyValueRetriever.tryRetrieve(
              env,
              new DefaultDependOnFutureShim(env),
              analysisCachingDeps.getObjectCodecs(),
              analysisCachingDeps.getFingerprintValueService(),
              analysisCachingDeps.getAnalysisCacheClient(),
              key,
              state,
              /* frontierNodeVersion= */ analysisCachingDeps.getSkyValueVersion());
      analysisCachingDeps.recordRetrievalResult(retrievalResult, key);
    } catch (SerializationException e) {
      // TODO: b/445242928 - also log this in BEP
      //
      // Don't crash the build if deserialization failed. Gracefully fallback to local evaluation.
      analysisCachingDeps.recordSerializationException(e, key);
      exception = e;
      retrievalResult = new NoCachedData(e.getReason());
    } catch (RuntimeException | InterruptedException e) {
      exception = e;
      throw e;
    } finally {
      if (retrievalResult == Restart.RESTART) {
        state.addRestart();
      } else if (analysisCachingDeps.getJsonLogWriter() != null && !state.isLogged()) {
        RemoteAnalysisJsonLogWriter logWriter = analysisCachingDeps.getJsonLogWriter();
        try (var entry = logWriter.startEntry("retrieve")) {
          entry.addField("start", state.getStart());
          entry.addField("end", Instant.now());
          entry.addField("skyKey", key.toString());
          entry.addField("cacheKey", base16().lowerCase().encode(state.getCacheKey().toBytes()));
          entry.addField("restarts", state.getRestarts());
          if (retrievalResult != null) {
            entry.addField("result", retrievalResult.toString());
          }
          if (exception != null) {
            entry.addField("exception", exception.getMessage());
          }
        }

        state.setLogged();
      }
    }

    if (retrievalResult instanceof RetrievedValue(SkyValue v)
        && !(v instanceof DeserializedSkyValue)) {
      throw new IllegalStateException(
          "deserialized SkyValue of type "
              + v.getClass().getCanonicalName()
              + " does not implement DeserializedSkyValue. Try using"
              + " @AutoCodec(deserializedInterface = DeserializedSkyValue.class)");
    }

    return retrievalResult;
  }

  private SkyValueRetrieverUtils() {}
}
