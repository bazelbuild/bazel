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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.util.concurrent.Futures.getDone;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Verify;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore.MissingFingerprintValueException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.StateEvictedException;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheClient;
import com.google.devtools.build.lib.skyframe.serialization.proto.DataType;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Fetches remotely stored {@link SkyValue}s by {@link SkyKey}. */
public final class SkyValueRetriever {

  /**
   * A wrapper for the mutable state of the analysis cache deserialization machinery.
   *
   * <p>It's mostly a continuation but also contains various kinds of data mostly useful for
   * debugging that are orthogonal to the continuation.
   */
  public static final class RetrievalContext {
    private SerializationState state;
    private int restarts;
    private boolean logged;
    @Nullable private Instant start;
    @Nullable private PackedFingerprint cacheKey;

    public RetrievalContext() {
      state = InitialQuery.INITIAL_QUERY;
      restarts = 0;
      logged = false;
      start = null;
      cacheKey = null;
    }

    public SerializationState getState() {
      return state;
    }

    public void setState(SerializationState newState) {
      state = newState;
    }

    public int getRestarts() {
      return restarts;
    }

    public void addRestart() {
      restarts++;
    }

    public boolean isLogged() {
      return logged;
    }

    public void setLogged() {
      Verify.verify(!logged);
      logged = true;
    }

    public Instant getStart() {
      return start;
    }

    public void setStart(Instant start) {
      Verify.verify(this.start == null);
      this.start = start;
    }

    public PackedFingerprint getCacheKey() {
      return cacheKey;
    }

    public void setCacheKey(PackedFingerprint cacheKey) {
      Verify.verify(this.cacheKey == null);
      this.cacheKey = cacheKey;
    }
  }

  /**
   * A {@link SkyKeyComputeState} implementation may additionally support this interface to enable
   * the {@link SkyFunction} to use {@link #tryRetrieve}.
   *
   * <p>Provides access to a {@link SerializationState} that is intended to persist across Skyframe
   * restarts and which is used to represent the state of the serialization machinery.
   */
  public interface RetrievalContextProvider {
    RetrievalContext getRetrievalContext();
  }

  /** A {@link RetrievalContextProvider} implemented as a {@link SkyKeyComputeState}. */
  public interface SerializableSkyKeyComputeState
      extends RetrievalContextProvider, SkyKeyComputeState {
    @Override
    default void close() {
      getRetrievalContext().getState().cleanupSerializationState();
    }
  }

  /**
   * Opaque state of serialization.
   *
   * <p>Clients must call {@link #cleanupSerializationState()} if state is evicted.
   *
   * <p>Each permitted type corresponds to a mostly sequential state.
   *
   * <ol>
   *   <li>{@link InitialQuery}: serializes the key and initiates fetching from {@link
   *       FingerprintValueService}, or via AnalysisCacheService client, if provided.
   *   <li>{@link WaitingForFutureValueBytes}: waits for value bytes from the {@link
   *       FingerprintValueService} to become available and skips over bytes associated with
   *       invalidation data. the result is available immediately, may directly transition to {@link
   *       RetrievedValue}.
   *   <li>{@link WaitingForCacheServiceResponse}: waits for value bytes from the
   *       AnalysisCacheService to become available.
   *   <li>{@link ProcessValueBytes}: a transient state that begins deserialization of the value
   *       bytes. When the result is available immediately, may directly transition to {@link
   *       RetrievedValue}.
   *   <li>{@link WaitingForFutureLookupContinuation}: waits for the {@link
   *       SkyframeLookupContinuation} to become available. This corresponds to immediate
   *       deserialization of any owned shared bytes. Immediate means that the shared bytes have
   *       been processed, but pending futures might be registered by this step. The significance of
   *       this instant is that all {@link SkyKey}s that need to be looked up to deserialize the
   *       value are known.
   *   <li>{@link WaitingForLookupContinuation}: looks up any {@link SkyKey}s needed for
   *       deserialization and waits for any resulting Skyframe restarts.
   *   <li>{@link WaitingForFutureResult}: waits for remaining futures needed to complete
   *       deserialization. This includes: any required <i>unowned</i> shared values; a small amount
   *       of work to set values in parents; and propagating the corresponding futures.
   *   <li>{@link NoCachedData}: a sentinel state representing a known cache miss. This is a
   *       terminal serialization state so that Skyframe restarts during the fallback local
   *       evaluation will not try to fetch the non-existent value again and waste computation.
   *   <li>{@link RetrievedValue}: returns the retrieved value, idempotently.
   * </ol>
   */
  public sealed interface SerializationState
      permits SkyValueRetriever.InitialQuery,
          SkyValueRetriever.WaitingForFutureValueBytes,
          SkyValueRetriever.WaitingForCacheServiceResponse,
          SkyValueRetriever.ProcessValueBytes,
          SkyValueRetriever.WaitingForFutureLookupContinuation,
          SkyValueRetriever.WaitingForLookupContinuation,
          SkyValueRetriever.WaitingForFutureResult,
          SkyValueRetriever.NoCachedData,
          SkyValueRetriever.RetrievedValue {
    default void cleanupSerializationState() {}
  }

  /** Return value of {@link #tryRetrieve}. */
  public sealed interface RetrievalResult
      permits SkyValueRetriever.Restart,
          SkyValueRetriever.NoCachedData,
          SkyValueRetriever.RetrievedValue {}

  /**
   * Deserialization requires a Skyframe restart to proceed.
   *
   * <p>This may indicate that deserialization is blocked on an unavailable Skyframe value or
   * asynchronous I/O. When {@link #tryRetrieve} is called inside a {@link SkyFunction} returning
   * null from {@link SkyFunction#compute} is appropriate.
   */
  public enum Restart implements RetrievalResult {
    RESTART;
  }

  /** The reason why a Skycache lookup rjkySesulted in a miss. */
  public enum CacheMissReason {
    UNKNOWN, // We don't know why
    SKYVALUE_MISS, // The fingerprint value store doesn't contain the requested SkyValue
    REFERENCED_OBJECT_MISS, // An object the requested SkyValue references is missing
    NOT_ATTEMPTED, // Skycache lookups are not possible due to the state Bazel is in
  }

  /**
   * There was no associated data in the cache for the requested key.
   *
   * <p>A typical client falls back on local computation upon seeing this.
   */
  public record NoCachedData(CacheMissReason reason)
      implements SerializationState, RetrievalResult {}

  /** The value was successfully retrieved. */
  public record RetrievedValue(SkyValue value) implements SerializationState, RetrievalResult {}

  /**
   * Attempts to retrieve the value associated with {@code key} from {@code
   * fingerprintValueService}.
   *
   * <p>The key is formed by serializing the SkyKey, appending the version metadata, and
   * fingerprinting the result.
   *
   * @param analysisCacheClient client for querying the AnalysisCacheService. Uses frontier-based
   *     invalidation and fetches directly from the {@code fingerprintValueService} if null.
   * @return a {@link RetrievalResult} instance. This can be {@link NoCachedData} when there is no
   *     data associated with the given key.
   */
  public static RetrievalResult tryRetrieve(
      LookupEnvironment env,
      DependOnFutureShim futuresShim,
      ObjectCodecs codecs,
      FingerprintValueService fingerprintValueService,
      @Nullable RemoteAnalysisCacheClient analysisCacheClient,
      SkyKey key,
      RetrievalContext retrievalContext,
      FrontierNodeVersion frontierNodeVersion)
      throws InterruptedException, SerializationException {
    SerializationState serializationState = retrievalContext.getState();
    try {
      while (true) {
        switch (serializationState) {
          case InitialQuery unused:
            {
              PackedFingerprint cacheKey =
                  FingerprintValueService.computeFingerprint(
                      fingerprintValueService, codecs, key, frontierNodeVersion);
              retrievalContext.setStart(Instant.now());
              retrievalContext.setCacheKey(cacheKey);
              ListenableFuture<?> responseFuture;
              if (analysisCacheClient == null) {
                ListenableFuture<byte[]> futureValueBytes;
                try {
                  futureValueBytes = fingerprintValueService.get(cacheKey);
                } catch (IOException e) {
                  throw new SerializationException("key lookup failed for " + key, e);
                }
                serializationState = new WaitingForFutureValueBytes(futureValueBytes);
                responseFuture = futureValueBytes;
              } else {
                ListenableFuture<ByteString> futureResponseBytes =
                    analysisCacheClient.lookup(ByteString.copyFrom(cacheKey.toBytes()));

                serializationState = new WaitingForCacheServiceResponse(futureResponseBytes);
                responseFuture = futureResponseBytes;
              }
              switch (futuresShim.dependOnFuture(responseFuture)) {
                case DONE:
                  break; // continues to the next state
                case NOT_DONE:
                  return Restart.RESTART;
              }
              break;
            }
          case WaitingForFutureValueBytes(ListenableFuture<byte[]> futureValueBytes):
            {
              byte[] valueBytes;
              try {
                valueBytes = getDone(futureValueBytes);
              } catch (ExecutionException e) {
                // This exception cannot happen in production code because the FingerprintValueStore
                // implementations used here don't throw
                Verify.verify(!(e.getCause() instanceof MissingFingerprintValueException));
                throw new SerializationException("getting value bytes for " + key, e);
              }
              if (valueBytes == null || valueBytes.length == 0) {
                // Serialized representations are never empty in this protocol. Some implementations
                // (not linked with Bazel) use empty bytes to indicate missing data
                serializationState = new NoCachedData(CacheMissReason.SKYVALUE_MISS);
                break;
              }
              var codedIn = CodedInputStream.newInstance(valueBytes);
              // Skips over the invalidation data key.
              //
              // TODO: b/364831651: consider removing this.
              try {
                int dataTypeOrdinal = codedIn.readInt32();
                switch (DataType.forNumber(dataTypeOrdinal)) {
                  case DATA_TYPE_EMPTY:
                    break;
                  case DATA_TYPE_FILE:
                  // fall through
                  case DATA_TYPE_LISTING:
                    {
                      var unusedKey = codedIn.readString();
                      break;
                    }
                  case DATA_TYPE_ANALYSIS_NODE, DATA_TYPE_EXECUTION_NODE:
                    {
                      var unusedKey = PackedFingerprint.readFrom(codedIn);
                      break;
                    }
                  default:
                    throw new SerializationException(
                        String.format(
                            "for key=%s, got unexpected data type with ordinal %d from value"
                                + " bytes=%s",
                            key, dataTypeOrdinal, HexFormat.of().formatHex(valueBytes)));
                }
              } catch (IOException e) {
                throw new SerializationException("Error parsing invalidation data key", e);
              }

              serializationState = new ProcessValueCodedInput(codedIn);
              break;
            }
          case WaitingForCacheServiceResponse(ListenableFuture<ByteString> futureBytes):
            {
              ByteString responseBytes;
              try {
                responseBytes = getDone(futureBytes);
              } catch (ExecutionException e) {
                throw new SerializationException("getting cache response for " + key, e);
              }
              if (responseBytes.isEmpty()) {
                serializationState = new NoCachedData(CacheMissReason.SKYVALUE_MISS);
                break;
              }

              serializationState = new ProcessValueByteString(responseBytes);
              break;
            }
          case ProcessValueBytes valueBytes:
            {
              Object value = valueBytes.deserializeWithSkyframe(codecs, fingerprintValueService);
              if (!(value instanceof ListenableFuture)) {
                serializationState = new RetrievedValue((SkyValue) value);
                break;
              }

              @SuppressWarnings("unchecked")
              var futureContinuation = (ListenableFuture<SkyframeLookupContinuation>) value;
              serializationState = new WaitingForFutureLookupContinuation(futureContinuation);
              switch (futuresShim.dependOnFuture(futureContinuation)) {
                case DONE:
                  break; // continues to the next state
                case NOT_DONE:
                  return Restart.RESTART;
              }
              break;
            }
          case WaitingForFutureLookupContinuation(
              ListenableFuture<SkyframeLookupContinuation> futureContinuation):
            // This state is transient. It discards the wrapping future before
            // WaitingForLookupContinuation so restarts from that state do not need repeat the
            // unwrapping.
            try {
              serializationState = new WaitingForLookupContinuation(getDone(futureContinuation));
            } catch (ExecutionException e) {
              CacheMissReason reason =
                  e.getCause() instanceof SerializationException se
                      ? se.getReason()
                      : CacheMissReason.UNKNOWN;
              throw new SerializationException(
                  "waiting for all owned shared values for " + key, e, reason);
            }
            break;
          case WaitingForLookupContinuation(SkyframeLookupContinuation lookupContinuation):
            {
              ListenableFuture<?> futureResult;
              try {
                futureResult =
                    lookupContinuation.process(env); // only source of InterruptedException
              } catch (SkyframeDependencyException e) {
                throw new SerializationException(
                    "skyframe dependency error during deserialization for " + key, e);
              }
              if (futureResult == null) {
                return Restart.RESTART;
              }
              serializationState = new WaitingForFutureResult(futureResult);
              switch (futuresShim.dependOnFuture(futureResult)) {
                case DONE:
                  break; // continues to the next state
                case NOT_DONE:
                  return Restart.RESTART;
              }
              break;
            }
          case WaitingForFutureResult(ListenableFuture<?> futureResult):
            try {
              serializationState = new RetrievedValue((SkyValue) getDone(futureResult));
            } catch (ExecutionException e) {
              throw new SerializationException("waiting for deserialization result for " + key, e);
            }
            break;
          case RetrievedValue value:
            return value;
          case NoCachedData noCachedData:
            return noCachedData;
        }
      }
    } catch (CancellationException e) {
      // CancellationException may be thrown from any of the calls to getDone. Reports
      // NO_CACHED_DATA to bail out of remote retrieval.
      //
      // TODO: b/438142239 - ideally, CancellationException would be handled by Skyframe. However,
      // it is only thrown by this method and NO_CACHED_DATA is a safe fallback.
      var result = new NoCachedData(CacheMissReason.UNKNOWN);
      serializationState = result;
      return result;
    } finally {
      retrievalContext.setState(serializationState);
    }
  }

  private enum InitialQuery implements SerializationState {
    INITIAL_QUERY;
  }

  @VisibleForTesting
  record WaitingForFutureValueBytes(ListenableFuture<byte[]> futureValueBytes)
      implements SerializationState {}

  @VisibleForTesting
  record WaitingForCacheServiceResponse(ListenableFuture<ByteString> cacheResponseBytes)
      implements SerializationState {}

  private static sealed interface ProcessValueBytes extends SerializationState
      permits ProcessValueCodedInput, ProcessValueByteString {
    Object deserializeWithSkyframe(
        ObjectCodecs codecs, FingerprintValueService fingerprintValueService)
        throws SerializationException;
  }

  private record ProcessValueCodedInput(CodedInputStream codedIn) implements ProcessValueBytes {
    @Override
    public Object deserializeWithSkyframe(
        ObjectCodecs codecs, FingerprintValueService fingerprintValueService)
        throws SerializationException {
      return codecs.deserializeWithSkyframe(fingerprintValueService, codedIn);
    }
  }

  private record ProcessValueByteString(ByteString valueBytes) implements ProcessValueBytes {
    @Override
    public Object deserializeWithSkyframe(
        ObjectCodecs codecs, FingerprintValueService fingerprintValueService)
        throws SerializationException {
      return codecs.deserializeWithSkyframe(fingerprintValueService, valueBytes);
    }
  }

  @VisibleForTesting
  record WaitingForFutureLookupContinuation(
      ListenableFuture<SkyframeLookupContinuation> futureContinuation)
      implements SerializationState {}

  @VisibleForTesting
  record WaitingForLookupContinuation(SkyframeLookupContinuation continuation)
      implements SerializationState {
    @Override
    public void cleanupSerializationState() {
      continuation.abandon(new StateEvictedException());
    }
  }

  @VisibleForTesting
  record WaitingForFutureResult(ListenableFuture<?> futureResult) implements SerializationState {}

  private SkyValueRetriever() {}
}
