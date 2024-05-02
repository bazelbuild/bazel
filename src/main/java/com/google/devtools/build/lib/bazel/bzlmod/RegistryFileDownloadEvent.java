package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
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

  static ImmutableMap<String, Optional<Checksum>> collectToMap(
      Collection<Postable> postables) {
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
          RepositoryCache.KeyType.SHA256, Hashing.sha256().hashBytes(bytes).toString());
    } catch (Checksum.InvalidChecksumException e) {
      // This can't happen since HashCode.toString() always returns a valid hash.
      throw new IllegalStateException(e);
    }
  }
}
