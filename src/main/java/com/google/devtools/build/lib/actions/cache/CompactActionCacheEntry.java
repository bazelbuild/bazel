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
package com.google.devtools.build.lib.actions.cache;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.cache.ActionCache.Entry;
import com.google.devtools.build.lib.util.StringIndexer;
import com.google.devtools.build.lib.util.VarInt;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Implements compact action cache entry that relies on the external index to store
 * file names.
 */
class CompactActionCacheEntry implements Entry {
  private final String actionKey;
  private final List<String> files;
  private Map<String, Metadata> mdMap;
  private Digest digest;

  static final CompactActionCacheEntry CORRUPTED = new CompactActionCacheEntry(null);

  /**
   * Creates new action cache entry and sets provided action key string.
   * Should be followed by one or more addFile() calls and, finally, a pack()
   * call.
   */
  CompactActionCacheEntry(String key) {
    actionKey = key;
    files = Lists.newArrayList();
    mdMap = Maps.newHashMap();
  }

  /**
   * Creates new action cache entry using given compressed entry data. Data
   * will stay in the compressed format until entry is actually used by the
   * dependency checker.
   */
  CompactActionCacheEntry(StringIndexer indexer, byte[] data) throws IOException {
    try {
      ByteBuffer source = ByteBuffer.wrap(data);

      byte[] actionKeyBytes = new byte[VarInt.getVarInt(source)];
      source.get(actionKeyBytes);
      actionKey = new String(actionKeyBytes, ISO_8859_1);

      digest = Digest.read(source);

      int count = VarInt.getVarInt(source);
      ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
      for (int i = 0; i < count; i++) {
        int id = VarInt.getVarInt(source);
        String filename = (id >= 0 ? indexer.getStringForIndex(id) : null);
        if (filename == null) {
          throw new IOException("Corrupted file index");
        }
        builder.add(filename);
      }
      if (source.remaining() > 0) {
        throw new IOException("serialized entry data has not been fully decoded");
      }
      files = builder.build();
    } catch (BufferUnderflowException e) {
      throw new IOException("encoded entry data is incomplete", e);
    }
  }

  /**
   * @return action data encoded as a byte[] array.
   */
  byte[] getData(StringIndexer indexer) {
    Preconditions.checkState(mdMap == null);
    Preconditions.checkState(!isCorrupted());

    try {
      byte[] actionKeyBytes = actionKey.getBytes(ISO_8859_1);

      // Estimate the size of the buffer:
      //   5 bytes max for the actionKey length
      // + the actionKey itself
      // + 16 bytes for the digest
      // + 5 bytes max for the file list length
      // + 5 bytes max for each file id
      int maxSize = VarInt.MAX_VARINT_SIZE + actionKeyBytes.length + Digest.MD5_SIZE
          + VarInt.MAX_VARINT_SIZE + files.size() * VarInt.MAX_VARINT_SIZE;
      ByteArrayOutputStream sink = new ByteArrayOutputStream(maxSize);

      VarInt.putVarInt(actionKeyBytes.length, sink);
      sink.write(actionKeyBytes);

      getFileDigest().write(sink);

      VarInt.putVarInt(files.size(), sink);
      for (String file : files) {
        VarInt.putVarInt(indexer.getOrCreateIndex(file), sink);
      }
      return sink.toByteArray();
    } catch (IOException e) {
      // This Exception can never be thrown by ByteArrayOutputStream.
      throw new AssertionError(e);
    }
  }

  @Override
  public String getActionKey() {
    return actionKey;
  }

  @Override
  public Digest getFileDigest() {
    if (digest == null) {
      digest = Digest.fromMetadata(mdMap);
      mdMap = null;
    }
    return digest;
  }

  @Override
  public void addFile(PathFragment relativePath, Metadata md) {
    // TODO(bazel-team): Refactor into addFiles(Map<PathFragment, Metadata>) which
    // will be called only once. This would allows us to get rid of the mdMap
    // used as temporary scratch space until the digest is computed.
    Preconditions.checkState(mdMap != null);
    Preconditions.checkState(!isCorrupted());
    Preconditions.checkState(digest == null);

    String execPath = relativePath.getPathString();
    files.add(execPath);
    mdMap.put(execPath, md);
  }

  @Override
  public Collection<String> getPaths() {
    return files;
  }

  @Override
  public boolean isCorrupted() {
    return actionKey == null;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("      actionKey = ").append(actionKey).append("\n");
    builder.append("      digestKey = ");
    if (digest == null) {
      builder.append(Digest.fromMetadata(mdMap)).append(" (from mdMap)\n");
    } else {
      builder.append(digest).append("\n");
    }
    List<String> fileInfo = Lists.newArrayListWithCapacity(files.size());
    fileInfo.addAll(files);
    Collections.sort(fileInfo);
    for (String info : fileInfo) {
      builder.append("      ").append(info).append("\n");
    }
    return builder.toString();
  }
}
