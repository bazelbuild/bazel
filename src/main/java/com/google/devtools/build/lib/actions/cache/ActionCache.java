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

package com.google.devtools.build.lib.actions.cache;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static java.util.Objects.requireNonNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissReason;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.io.PrintStream;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * An interface defining a cache of already-executed Actions.
 *
 * <p>The name of this class is misleading; it doesn't cache the actual actions, only a fingerprint
 * of all action properties that matter for cache invalidation (action key, path and contents of
 * input and outputs files, environment variables, execution properties, and certain flags), so we
 * can tell if we need to rerun an action given the current state of the file system.
 *
 * <p>Each action entry uses the path of the action's primary output as the key.
 */
@ThreadCompatible
public interface ActionCache {

  /** Updates the cache entry for the specified key. */
  void put(String key, ActionCache.Entry entry);

  /**
   * Returns the cache entry for the specified key, or null if not found.
   *
   * <p>If an entry exists but is corrupted, returns {@link ActionCache.Entry.CORRUPTED}. Callers
   * should check {@link ActionCache.Entry#isCorrupted()} before inspecting anything else on the
   * entry.
   */
  @Nullable
  ActionCache.Entry get(String key);

  /** Removes entry from cache */
  void remove(String key);

  /** Removes entry from cache that matches the predicate. */
  void removeIf(Predicate<ActionCache.Entry> predicate);

  /** An action cache entry. */
  public static final class Entry {
    /** Unique instance standing for a corrupted cache entry. */
    public static final ActionCache.Entry CORRUPTED =
        new Entry(null, null, ImmutableMap.of(), ImmutableMap.of());

    // Digest of all relevant properties of the action for cache invalidation purposes.
    // Null if the entry is corrupted.
    @Nullable private final byte[] digest;

    // List of input paths discovered by the action.
    // Null if the action does not discover inputs.
    @Nullable private final ImmutableList<String> discoveredInputPaths;

    // Output metadata.
    // Only present when building without the bytes, and even then, only for remotely stored files.
    private final ImmutableMap<String, FileArtifactValue> outputFileMetadata;
    private final ImmutableMap<String, SerializableTreeArtifactValue> outputTreeMetadata;

    Entry(
        @Nullable byte[] digest,
        @Nullable ImmutableList<String> discoveredInputPaths,
        ImmutableMap<String, FileArtifactValue> outputFileMetadata,
        ImmutableMap<String, SerializableTreeArtifactValue> outputTreeMetadata) {
      this.digest = digest;
      this.discoveredInputPaths = discoveredInputPaths;
      this.outputFileMetadata = outputFileMetadata;
      this.outputTreeMetadata = outputTreeMetadata;
    }

    /** Returns whether this cache entry is corrupted and should be ignored. */
    public boolean isCorrupted() {
      return digest == null;
    }

    /**
     * Returns a digest encoding all relevant properties of the action for cache invalidation
     * purposes.
     */
    public byte[] getDigest() {
      checkState(!isCorrupted());
      return digest;
    }

    /** Returns whether the action discovers inputs. */
    public boolean discoversInputs() {
      checkState(!isCorrupted());
      return discoveredInputPaths != null;
    }

    /**
     * Returns the list of discovered input paths, or null if the action does not discover inputs.
     */
    @Nullable
    public ImmutableList<String> getDiscoveredInputPaths() {
      checkState(!isCorrupted());
      return discoveredInputPaths;
    }

    /** Gets the metadata of an output file. */
    @Nullable
    public FileArtifactValue getOutputFile(Artifact output) {
      checkState(!isCorrupted());
      return outputFileMetadata.get(output.getExecPathString());
    }

    /** Gets the metadata of all output files. */
    public ImmutableMap<String, FileArtifactValue> getOutputFiles() {
      checkState(!isCorrupted());
      return outputFileMetadata;
    }

    /** Gets the metadata of an output tree. */
    @Nullable
    public SerializableTreeArtifactValue getOutputTree(SpecialArtifact output) {
      checkState(!isCorrupted());
      return outputTreeMetadata.get(output.getExecPathString());
    }

    /** Gets the metadata of all output trees. */
    public ImmutableMap<String, SerializableTreeArtifactValue> getOutputTrees() {
      checkState(!isCorrupted());
      return outputTreeMetadata;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("digest", digest)
          .add("discoveredInputPaths", discoveredInputPaths)
          .add("outputFileMetadata", outputFileMetadata)
          .add("outputTreeMetadata", outputTreeMetadata)
          .toString();
    }

    void dump(PrintStream out) {
      if (isCorrupted()) {
        out.println("  CORRUPTED");
        return;
      }
      out.format("  digest = %s\n", formatDigest(digest));
      if (discoveredInputPaths != null) {
        out.println("  discoveredInputPaths =");
        for (String path : ImmutableList.sortedCopyOf(discoveredInputPaths)) {
          out.format("    %s\n", path);
        }
      }

      if (!outputFileMetadata.isEmpty()) {
        out.println("  outputFileMetadata =");
        for (String path : ImmutableList.sortedCopyOf(outputFileMetadata.keySet())) {
          out.format("    %s = %s\n", path, outputFileMetadata.get(path));
        }
      }

      if (!outputTreeMetadata.isEmpty()) {
        out.println("  outputTreeMetadata =");
        for (String path : ImmutableList.sortedCopyOf(outputTreeMetadata.keySet())) {
          out.format("    %s = %s\n", path, outputTreeMetadata.get(path));
        }
      }
    }

    private static String formatDigest(byte[] digest) {
      return BaseEncoding.base16().lowerCase().encode(digest);
    }

    /** Serializable representation of {@link TreeArtifactValue}. */
    public record SerializableTreeArtifactValue(
        ImmutableMap<String, FileArtifactValue> childValues,
        Optional<FileArtifactValue> archivedFileValue,
        Optional<PathFragment> resolvedPath) {
      public SerializableTreeArtifactValue {
        requireNonNull(childValues, "childValues");
        requireNonNull(archivedFileValue, "archivedFileValue");
        requireNonNull(resolvedPath, "resolvedPath");
      }

      public static SerializableTreeArtifactValue create(
          ImmutableMap<String, FileArtifactValue> childValues,
          Optional<FileArtifactValue> archivedFileValue,
          Optional<PathFragment> resolvedPath) {
        return new SerializableTreeArtifactValue(childValues, archivedFileValue, resolvedPath);
      }

      /**
       * Creates {@link SerializableTreeArtifactValue} from {@link TreeArtifactValue} by collecting
       * children and archived artifact which are remote.
       *
       * <p>If no remote value, {@link Optional#empty} is returned.
       */
      public static Optional<SerializableTreeArtifactValue> createSerializable(
          TreeArtifactValue treeMetadata) {
        ImmutableMap<String, FileArtifactValue> childValues =
            treeMetadata.getChildValues().entrySet().stream()
                // Only save remote tree file
                .filter(e -> e.getValue().isRemote())
                .collect(
                    toImmutableMap(e -> e.getKey().getTreeRelativePathString(), e -> e.getValue()));

        // Only save remote archived artifact
        Optional<FileArtifactValue> archivedFileValue =
            treeMetadata
                .getArchivedRepresentation()
                .filter(ar -> ar.archivedFileValue().isRemote())
                .map(ar -> ar.archivedFileValue());

        Optional<PathFragment> resolvedPath = treeMetadata.getResolvedPath();

        if (childValues.isEmpty() && archivedFileValue.isEmpty() && resolvedPath.isEmpty()) {
          return Optional.empty();
        }

        return Optional.of(
            SerializableTreeArtifactValue.create(childValues, archivedFileValue, resolvedPath));
      }
    }

    /** A builder for an action cache entry. */
    public static final class Builder {
      private final String actionKey;

      // Combined input and output metadata.
      private final HashMap<String, FileArtifactValue> metadataMap = new HashMap<>();

      private final ImmutableMap<String, String> clientEnv;
      private final ImmutableMap<String, String> execProperties;

      // Discovered inputs.
      // Null if the action does not discover inputs.
      @Nullable private final ImmutableList.Builder<String> discoveredInputPaths;

      private final ImmutableMap.Builder<String, FileArtifactValue> outputFileMetadata =
          ImmutableMap.builder();
      private final ImmutableMap.Builder<String, SerializableTreeArtifactValue> outputTreeMetadata =
          ImmutableMap.builder();

      // Settings that affect the outcome of an action but aren't captured in the file metadata.
      private final OutputPermissions outputPermissions;
      private final boolean useArchivedTreeArtifacts;

      /**
       * Creates a new builder.
       *
       * @param discoversInputs whether the action discovers inputs.
       * @param outputPermissions the requested output permissions.
       * @param useArchivedTreeArtifacts whether archived tree artifacts are enabled.
       */
      public Builder(
          String actionKey,
          boolean discoversInputs,
          ImmutableMap<String, String> clientEnv,
          ImmutableMap<String, String> execProperties,
          OutputPermissions outputPermissions,
          boolean useArchivedTreeArtifacts) {
        this.actionKey = actionKey;
        this.clientEnv = clientEnv;
        this.execProperties = execProperties;
        this.discoveredInputPaths = discoversInputs ? ImmutableList.builder() : null;
        this.outputPermissions = outputPermissions;
        this.useArchivedTreeArtifacts = useArchivedTreeArtifacts;
      }

      /** Adds metadata of an input file. */
      @CanIgnoreReturnValue
      public Builder addInputFile(Artifact artifact, FileArtifactValue metadata) {
        addInputFile(artifact, metadata, /* saveExecPath= */ false);
        return this;
      }

      /** Adds metadata of an input file. */
      @CanIgnoreReturnValue
      public Builder addInputFile(
          Artifact artifact, FileArtifactValue metadata, boolean saveExecPath) {
        String execPath = artifact.getExecPathString();
        if (discoveredInputPaths != null && saveExecPath) {
          discoveredInputPaths.add(execPath);
        }
        metadataMap.put(execPath, metadata);
        return this;
      }

      /** Adds an output file. */
      @CanIgnoreReturnValue
      public Builder addOutputFile(Artifact output, FileArtifactValue metadata) {
        return addOutputFile(output, metadata, /* saveFileMetadata= */ false);
      }

      /** Adds an output file. */
      @CanIgnoreReturnValue
      public Builder addOutputFile(
          Artifact output, FileArtifactValue metadata, boolean saveFileMetadata) {
        checkArgument(
            !output.isTreeArtifact() && !output.isChildOfDeclaredDirectory(),
            "Must use addOutputTree to save tree artifacts and their children: %s",
            output);
        String execPath = output.getExecPathString();
        // Only save remote file metadata
        if (saveFileMetadata && metadata.isRemote()) {
          outputFileMetadata.put(execPath, metadata);
        }
        metadataMap.put(execPath, metadata);
        return this;
      }

      /** Adds an output tree. */
      @CanIgnoreReturnValue
      public Builder addOutputTree(SpecialArtifact output, TreeArtifactValue metadata) {
        return addOutputTree(output, metadata, /* saveTreeMetadata= */ false);
      }

      /** Adds an output tree. */
      @CanIgnoreReturnValue
      public Builder addOutputTree(
          SpecialArtifact output, TreeArtifactValue metadata, boolean saveTreeMetadata) {
        checkArgument(output.isTreeArtifact(), "artifact must be a tree artifact: %s", output);
        String execPath = output.getExecPathString();
        if (saveTreeMetadata) {
          SerializableTreeArtifactValue.createSerializable(metadata)
              .ifPresent(value -> outputTreeMetadata.put(execPath, value));
        }
        metadataMap.put(execPath, metadata.getMetadata());
        return this;
      }

      public Entry build() {
        return new Entry(
            computeDigest(
                actionKey,
                discoveredInputPaths != null,
                metadataMap,
                clientEnv,
                execProperties,
                outputPermissions,
                useArchivedTreeArtifacts),
            discoveredInputPaths != null ? discoveredInputPaths.build() : null,
            outputFileMetadata.buildOrThrow(),
            outputTreeMetadata.buildOrThrow());
      }

      private static byte[] computeDigest(
          String actionKey,
          boolean discoversInputs,
          Map<String, FileArtifactValue> metadataMap,
          Map<String, String> clientEnv,
          Map<String, String> execProperties,
          OutputPermissions outputPermissions,
          boolean useArchivedTreeArtifacts) {
        Fingerprint fp = new Fingerprint();
        fp.addString(actionKey);
        fp.addBoolean(discoversInputs);
        fp.addBytes(MetadataDigestUtils.fromMetadata(metadataMap));
        fp.addBytes(computeMapDigest(clientEnv));
        fp.addBytes(computeMapDigest(execProperties));
        fp.addInt(outputPermissions.getPermissionsMode());
        fp.addBoolean(useArchivedTreeArtifacts);
        return fp.digestAndReset();
      }

      private static byte[] computeMapDigest(Map<String, String> map) {
        byte[] result = new byte[0];
        Fingerprint fp = new Fingerprint();
        for (Map.Entry<String, String> entry : map.entrySet()) {
          fp.addString(entry.getKey());
          fp.addString(entry.getValue());
          result = DigestUtils.combineUnordered(result, fp.digestAndReset());
        }
        return result;
      }
    }
  }

  /**
   * Give persistent cache implementations a notification to write to disk.
   *
   * @return size in bytes of the serialized cache.
   */
  long save() throws IOException;

  /** Clear the action cache, closing all opened file handle. */
  void clear();

  /**
   * Returns an {@link ActionCache} with the same backing directory, but whose contents may have
   * been garbage collected.
   *
   * <p>May be safely interrupted. Upon interruption, this instance, including its backing
   * directory, remains valid. Otherwise, the return value may be the current instance or a
   * different one, depending on whether garbage collection was deemed necessary. If a different
   * instance is returned, the current instance must not be used further. Thus, safe usage of this
   * method looks like {@code actionCache = actionCache.trim(threshold, maxAge)}.
   *
   * @param threshold the fraction of stale entries required to trigger garbage collection
   * @param maxAge the age at which entries are considered stale
   * @return either the current instance, or a fresh instance that replaces it
   * @throws IOException if an I/O error occurs
   * @throws InterruptedException in case of interruption
   */
  ActionCache trim(float threshold, Duration maxAge) throws IOException, InterruptedException;

  /** Dumps the action cache into a human-readable format. */
  void dump(PrintStream out);

  /** The number of entries in the cache. */
  int size();

  /** Accounts one cache hit. */
  void accountHit();

  /** Accounts one cache miss for the given reason. */
  void accountMiss(MissReason reason);

  /**
   * Populates the given builder with statistics.
   *
   * <p>The extracted values are not guaranteed to be a consistent snapshot of the metrics tracked
   * by the action cache. Therefore, even if it is safe to call this function at any point in time,
   * this should only be called once there are no actions running.
   */
  void mergeIntoActionCacheStatistics(ActionCacheStatistics.Builder builder);

  /** Resets the current statistics to zero. */
  void resetStatistics();

  /** Duration it took to load the action cache. Might be null if not loaded in this invocation. */
  @Nullable
  default Duration getLoadTime() {
    return null;
  }
}
