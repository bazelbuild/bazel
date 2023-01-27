package com.google.devtools.build.lib.vfs;

public interface LeaseService {
  boolean isAlive(byte[] digest, long size, int locationIndex);
}
