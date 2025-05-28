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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import java.util.Collection;
import java.util.Optional;

/** Event that records the fact that a file has been downloaded from a remote registry. */
public record RegistryFileDownloadEvent(String uri, Optional<Checksum> checksum)
    implements Postable {

  public static RegistryFileDownloadEvent create(String uri, Optional<byte[]> content) {
    return new RegistryFileDownloadEvent(uri, content.map(RegistryFileDownloadEvent::computeHash));
  }

  static ImmutableMap<String, Optional<Checksum>> collectToMap(Collection<Postable> postables) {
    ImmutableMap.Builder<String, Optional<Checksum>> builder = ImmutableMap.builder();
    for (Postable postable : postables) {
      if (postable instanceof RegistryFileDownloadEvent event) {
        builder.put(event.uri(), event.checksum());
      }
    }
    return builder.buildKeepingLast();
  }

  private static Checksum computeHash(byte[] bytes) {
    try {
      return Checksum.fromString(
          DownloadCache.KeyType.SHA256, Hashing.sha256().hashBytes(bytes).toString());
    } catch (Checksum.InvalidChecksumException e) {
      // This can't happen since HashCode.toString() always returns a valid hash.
      throw new IllegalStateException(e);
    }
  }
}
