package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.Optional;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
public class SuperChecksum {
  private Checksum checksum;

  private SuperChecksum(Checksum checksum) {
    this.checksum = checksum;
  }

  public Checksum getChecksum() {
    return checksum;
  }

  public static SuperChecksum fromString(KeyType keyType, String hash) {
    Checksum checksum = null;
    
    /* The if statement makes sure Checksum.fromString() will never return an exception. */
    try {
      if(keyType == KeyType.SHA256 && keyType.isValid(hash)) {
        checksum = Checksum.fromString(keyType, hash);
      }
    }
    catch(Checksum.InvalidChecksumException e) {
    }

    return new SuperChecksum(checksum);
  }
}