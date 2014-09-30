// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.exec;

import com.google.common.collect.Maps;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.ArtifactMetadataCache;
import com.google.protobuf.ByteString;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Legacy implementation of {@link ActionInputFileCache} that delegates to an
 * {@link ArtifactMetadataCache}. This is only used when Skyframe is not running the execution
 * phase.
 *
 * <p>We look up digests by first resolving relative paths to an Artifact, and then keying that
 * into the ArtifactMetadataCache.
 *
 * <p>Since not all remote action inputs are Artifacts, we use a {@link ActionInputFileCache}
 * delegate on fallback.
 */
public class LegacyActionInputFileCache implements ActionInputFileCache {

  private static final Logger LOG = Logger.getLogger(LegacyActionInputFileCache.class.getName());

  private final ArtifactMetadataCache artifactMetadataCache;
  private final Map<ByteString, ActionInput> digestToPathMap = Maps.newConcurrentMap();

  private final ActionInputFileCache fileCacheDelegate;
  private final File cwd;

  public LegacyActionInputFileCache(ArtifactMetadataCache artifactMetadataCache,
      ActionInputFileCache delegate, File cwd) {
    this.artifactMetadataCache = artifactMetadataCache;
    this.fileCacheDelegate = delegate;
    this.cwd = cwd;
  }

  @Nullable
  @Override
  public File getFileFromDigest(ByteString digest) throws IOException {
    LOG.fine("Looking up digest " + digest);
    ActionInput artifact = digestToPathMap.get(digest);
    if (artifact == null) {
      return fileCacheDelegate.getFileFromDigest(digest);
    }
    String relPath = artifact.getExecPathString();
    return relPath.startsWith("/") ? new File(relPath) : new File(cwd, relPath);
  }

  private Artifact getArtifactFromActionInput(ActionInput actionInput) {
    return (actionInput instanceof Artifact) ? (Artifact) actionInput : null;
  }

  @Override
  public long getSizeInBytes(ActionInput actionInput) throws IOException {
    Artifact artifact = getArtifactFromActionInput(actionInput);
    if (artifact == null) {
      return fileCacheDelegate.getSizeInBytes(actionInput);
    }
    return getSizeInBytes(artifact);
  }

  private long getSizeInBytes(Artifact artifact) throws IOException {
    return artifactMetadataCache.getSize(artifact);
  }

  @Override
  public ByteString getDigest(ActionInput actionInput) throws IOException {
    Artifact artifact = getArtifactFromActionInput(actionInput);
    if (artifact == null) {
      return fileCacheDelegate.getDigest(actionInput);
    }
    return getDigest(artifact);
  }

  private ByteString getDigest(Artifact artifact) throws IOException {
    byte[] rawDigest = artifactMetadataCache.getDigestMaybe(artifact);
    if (rawDigest == null) {
      return fileCacheDelegate.getDigest(artifact);
    }
    // Cache hit case.
    ByteString digest = ByteString.copyFrom(
        BaseEncoding.base16().lowerCase().encode(rawDigest).getBytes(StandardCharsets.US_ASCII));
    // Note that we needn't populate this mapping if already there.
    digestToPathMap.put(digest, artifact);

    return digest;
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    return digestToPathMap.containsKey(digest)
           || fileCacheDelegate.contentsAvailableLocally(digest);
  }
}
