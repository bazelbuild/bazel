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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Throwables.getRootCause;
import static com.google.common.util.concurrent.Futures.getDone;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.hash.HashCode;
import com.google.common.primitives.Bytes;
import com.google.common.primitives.Longs;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore.MissingFingerprintValueException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.StateEvictedException;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.ClientId.SnapshotClientId;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheClient;
import com.google.devtools.build.lib.skyframe.serialization.proto.DataType;
import com.google.devtools.build.skyframe.IntVersion;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/** Fetches remotely stored {@link SkyValue}s by {@link SkyKey}. */
public final class SkyValueRetriever {
  /** Clients of this class should use this value as the initial state. */
  public static final SerializationState INITIAL_STATE = InitialQuery.INITIAL_QUERY;

  /**
   * A {@link SkyKeyComputeState} implementation may additionally support this interface to enable
   * the {@link SkyFunction} to use {@link #tryRetrieve}.
   *
   * <p>Provides access to a {@link SerializationState} that is intended to persist across Skyframe
   * restarts. Implementations should initialize the value to {@link #INITIAL_STATE}.
   */
  public interface SerializationStateProvider {
    SerializationState getSerializationState();

    void setSerializationState(SerializationState state);
  }

  /** A {@link SerializationStateProvider} implemented as a {@link SkyKeyComputeState}. */
  public interface SerializableSkyKeyComputeState
      extends SerializationStateProvider, SkyKeyComputeState {
    @Override
    default void close() {
      getSerializationState().cleanupSerializationState();
    }
  }

  /** Returned status of {@link DependOnFutureShim#dependOnFuture}. */
  public enum ObservedFutureStatus {
    /** If the future was already done. */
    DONE,
    /**
     * If the future was not done.
     *
     * <p>Indicates that a Skyframe restart is needed.
     */
    NOT_DONE
  }

  /**
   * Encapsulates {@link Environment#dependOnFuture}, {@link Environment#valuesMissing} in a single
   * method.
   *
   * <p>Decoupling this from {@link Environment} simplifies testing and the API.
   */
  public interface DependOnFutureShim {
    /**
     * Outside of testing, delegates to {@link Environment#dependOnFuture} then {@link
     * Environment#valuesMissing} to determine the return value.
     *
     * @throws IllegalStateException if called when an underlying environment's {@link
     *     Environment#valuesMissing} is already true.
     */
    ObservedFutureStatus dependOnFuture(ListenableFuture<?> future);
  }

  /** A convenience implementation used with an {@link Environment} instance. */
  public static final class DefaultDependOnFutureShim implements DependOnFutureShim {
    private final Environment env;

    public DefaultDependOnFutureShim(Environment env) {
      this.env = env;
    }

    @Override
    public ObservedFutureStatus dependOnFuture(ListenableFuture<?> future) {
      checkState(!env.valuesMissing());
      env.dependOnFuture(future);
      return env.valuesMissing() ? ObservedFutureStatus.NOT_DONE : ObservedFutureStatus.DONE;
    }
  }

  /**
   * Opaque state of serialization.
   *
   * <p>Clients must call {@link cleanupSerializationState} if state is evicted.
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

  /**
   * There was no associated data in the cache for the requested key.
   *
   * <p>A typical client falls back on local computation upon seeing this.
   */
  public enum NoCachedData implements SerializationState, RetrievalResult {
    NO_CACHED_DATA;
  }

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
      SerializationStateProvider stateProvider,
      FrontierNodeVersion frontierNodeVersion)
      throws InterruptedException, SerializationException {
    SerializationState nextState = stateProvider.getSerializationState();
    try {
      while (true) {
        switch (nextState) {
          case InitialQuery unused:
            {
              PackedFingerprint cacheKey =
                  SkyKeySerializationHelper.computeFingerprint(
                      codecs, fingerprintValueService, key, frontierNodeVersion);

              ListenableFuture<?> responseFuture;
              if (analysisCacheClient == null) {
                ListenableFuture<byte[]> futureValueBytes;
                try {
                  futureValueBytes = fingerprintValueService.get(cacheKey);
                } catch (IOException e) {
                  throw new SerializationException("key lookup failed for " + key, e);
                }
                nextState = new WaitingForFutureValueBytes(futureValueBytes);
                responseFuture = futureValueBytes;
              } else {
                ListenableFuture<ByteString> futureResponseBytes =
                    analysisCacheClient.lookup(ByteString.copyFrom(cacheKey.toBytes()));

                nextState = new WaitingForCacheServiceResponse(futureResponseBytes);
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
                if (getRootCause(e) instanceof MissingFingerprintValueException) {
                  nextState = NoCachedData.NO_CACHED_DATA;
                  break;
                }
                throw new SerializationException("getting value bytes for " + key, e);
              }
              if (valueBytes.length == 0) {
                // Serialized representations are never empty in this protocol. Some implementations
                // use empty bytes to indicate missing data.
                nextState = NoCachedData.NO_CACHED_DATA;
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

              nextState = new ProcessValueCodedInput(codedIn);
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
                nextState = NoCachedData.NO_CACHED_DATA;
                break;
              }

              nextState = new ProcessValueByteString(responseBytes);
              break;
            }
          case ProcessValueBytes valueBytes:
            {
              Object value = valueBytes.deserializeWithSkyframe(codecs, fingerprintValueService);
              if (!(value instanceof ListenableFuture)) {
                nextState = new RetrievedValue((SkyValue) value);
                break;
              }

              @SuppressWarnings("unchecked")
              var futureContinuation = (ListenableFuture<SkyframeLookupContinuation>) value;
              nextState = new WaitingForFutureLookupContinuation(futureContinuation);
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
              nextState = new WaitingForLookupContinuation(getDone(futureContinuation));
            } catch (ExecutionException e) {
              throw new SerializationException("waiting for all owned shared values for " + key, e);
            }
            break;
          case WaitingForLookupContinuation(SkyframeLookupContinuation continuation):
            {
              ListenableFuture<?> futureResult;
              try {
                futureResult = continuation.process(env); // only source of InterruptedException
              } catch (SkyframeDependencyException e) {
                throw new SerializationException(
                    "skyframe dependency error during deserialization for " + key, e);
              }
              if (futureResult == null) {
                return Restart.RESTART;
              }
              nextState = new WaitingForFutureResult(futureResult);
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
              nextState = new RetrievedValue((SkyValue) getDone(futureResult));
            } catch (ExecutionException e) {
              throw new SerializationException("waiting for deserialization result for " + key, e);
            }
            break;
          case RetrievedValue value:
            return value;
          case NoCachedData unused:
            return NoCachedData.NO_CACHED_DATA;
        }
      }
    } finally {
      stateProvider.setSerializationState(nextState);
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

  /** A tuple representing the version of a cached SkyValue in the frontier. */
  public static final class FrontierNodeVersion {
    public static final FrontierNodeVersion CONSTANT_FOR_TESTING =
        new FrontierNodeVersion(
            "123",
            HashCode.fromInt(42),
            IntVersion.of(9000),
            "distinguisher",
            /* useFakeStampData= */ true,
            Optional.of(new SnapshotClientId("for_testing", 123)));

    // Fingerprints of version components.
    private final byte[] topLevelConfigFingerprint;
    private final byte[] blazeInstallMD5Fingerprint;
    private final byte[] evaluatingVersionFingerprint;

    // Fingerprint of the distinguisher for allowing test cases to share a
    // static cache.
    private final byte[] distinguisherBytesForTesting;

    private final boolean useFakeStampData;

    // Fingerprint of the full version.
    private final byte[] precomputedFingerprint;

    private final Optional<ClientId> clientId;

    public FrontierNodeVersion(
        String topLevelConfigChecksum,
        HashCode blazeInstallMD5,
        IntVersion evaluatingVersion,
        String distinguisherBytesForTesting,
        boolean useFakeStampData,
        Optional<ClientId> clientId) {
      this.topLevelConfigFingerprint = topLevelConfigChecksum.getBytes(UTF_8);
      this.blazeInstallMD5Fingerprint = blazeInstallMD5.asBytes();
      this.evaluatingVersionFingerprint = Longs.toByteArray(evaluatingVersion.getVal());
      this.distinguisherBytesForTesting = distinguisherBytesForTesting.getBytes(UTF_8);
      this.useFakeStampData = useFakeStampData;
      this.precomputedFingerprint =
          Bytes.concat(
              this.topLevelConfigFingerprint,
              this.blazeInstallMD5Fingerprint,
              this.evaluatingVersionFingerprint,
              this.distinguisherBytesForTesting,
              this.useFakeStampData ? new byte[] {1} : new byte[] {0});

      // This is undigested.
      this.clientId = clientId;
    }

    /**
     * Returns the snapshot of the workspace.
     *
     * <p>Can be empty if snapshots are not supported by the workspace.
     */
    @SuppressWarnings("unused") // to be integrated
    public Optional<ClientId> getClientId() {
      return clientId;
    }

    public byte[] getTopLevelConfigFingerprint() {
      return topLevelConfigFingerprint;
    }

    public byte[] getPrecomputedFingerprint() {
      return precomputedFingerprint;
    }

    public byte[] concat(byte[] input) {
      return Bytes.concat(precomputedFingerprint, input);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("topLevelConfig", Arrays.hashCode(topLevelConfigFingerprint))
          .add("blazeInstall", Arrays.hashCode(blazeInstallMD5Fingerprint))
          .add("evaluatingVersion", Arrays.hashCode(evaluatingVersionFingerprint))
          .add("distinguisherBytesForTesting", Arrays.hashCode(distinguisherBytesForTesting))
          .add("useFakeStampData", useFakeStampData)
          .add("precomputed", hashCode())
          .toString();
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(precomputedFingerprint);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof FrontierNodeVersion that)) {
        return false;
      }
      return Arrays.equals(precomputedFingerprint, that.precomputedFingerprint);
    }
  }
}
