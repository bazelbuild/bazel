// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.bazel;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.google.devtools.build.lib.jni.JniLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.security.DigestException;
import java.security.MessageDigest;
import java.util.Objects;

/** A {@link MessageDigest} for BLAKE3. */
@SuppressWarnings("restricted") // Uses restricted java.lang.foreign methods.
public final class Blake3MessageDigest extends MessageDigest {
  // These constants match the native definitions in:
  // https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3.h
  public static final int KEY_LEN = 32;
  public static final int OUT_LEN = 32;

  static {
    JniLoader.loadJni();
  }

  // All functions are bound with Linker.Option.critical, which avoids the
  // overhead of a full native call transition. This is safe as they neither
  // block nor call back into Java. Passing allowHeapAccess as true makes it
  // possible to pass heap byte arrays to the native functions directly,
  // without copying. A critical call delays safepoints until it returns, so
  // each call hashes at most MAX_CRITICAL_INPUT_LENGTH bytes.
  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup LOOKUP = SymbolLookup.loaderLookup();
  private static final MethodHandle HASHER_SIZE =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hasher_size"),
          FunctionDescriptor.of(JAVA_INT),
          Linker.Option.critical(/* allowHeapAccess= */ false));
  private static final MethodHandle HASHER_INIT =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hasher_init"),
          FunctionDescriptor.ofVoid(ADDRESS),
          Linker.Option.critical(/* allowHeapAccess= */ false));
  private static final MethodHandle HASHER_UPDATE =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hasher_update"),
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG),
          Linker.Option.critical(/* allowHeapAccess= */ true));
  private static final MethodHandle HASHER_FINALIZE =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hasher_finalize"),
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG),
          Linker.Option.critical(/* allowHeapAccess= */ true));
  private static final MethodHandle HASH =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hash"),
          FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS),
          Linker.Option.critical(/* allowHeapAccess= */ true));

  // Hashing this many bytes takes tens of microseconds. It is a multiple of the
  // input size that the widest SIMD implementations process in parallel.
  private static final int MAX_CRITICAL_INPUT_LENGTH = 64 * 1024;

  /**
   * Looks up a native symbol, throwing {@link UnsatisfiedLinkError} just like a JNI method would
   * if it is missing. {@link BazelHashFunctions} relies on this error to detect that the native
   * library was built without the blake3 symbols when bootstrapping Bazel.
   */
  private static MemorySegment findSymbol(String name) {
    return LOOKUP
        .find(name)
        .orElseThrow(() -> new UnsatisfiedLinkError("undefined symbol: " + name));
  }

  private static final int STATE_SIZE;

  static {
    try {
      STATE_SIZE = (int) HASHER_SIZE.invokeExact();
    } catch (Throwable t) {
      throw new ExceptionInInitializerError(t);
    }
  }

  // The native hasher state. blake3_hasher contains uint64_t fields, so it
  // requires 8-byte alignment. A confined arena is cheap to allocate and free,
  // but restricts the instance to the thread that created it. Instances that
  // are never closed, such as those created via MessageDigest.getInstance,
  // leak their state.
  private final Arena arena = Arena.ofConfined();
  private final MemorySegment hasher = arena.allocate(STATE_SIZE, 8);
  private final byte[] oneByteArray = new byte[1];

  public Blake3MessageDigest() {
    super("BLAKE3");
    engineReset();
  }

  /** Returns the BLAKE3 digest of the given range of bytes. */
  static byte[] hash(byte[] data, int offset, int length) {
    if (length > MAX_CRITICAL_INPUT_LENGTH) {
      Blake3MessageDigest messageDigest = new Blake3MessageDigest();
      try {
        messageDigest.engineUpdate(data, offset, length);
        return messageDigest.engineDigest();
      } finally {
        messageDigest.close();
      }
    }
    MemorySegment input = MemorySegment.ofArray(data).asSlice(offset, length);
    byte[] out = new byte[OUT_LEN];
    try {
      HASH.invokeExact(input, (long) length, MemorySegment.ofArray(out));
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
    return out;
  }

  @Override
  public void engineUpdate(byte[] data, int offset, int length) {
    Objects.checkFromIndexSize(offset, length, data.length);
    MemorySegment input = MemorySegment.ofArray(data).asSlice(offset, length);
    for (long start = 0; start < length; start += MAX_CRITICAL_INPUT_LENGTH) {
      long sliceLength = Math.min(length - start, MAX_CRITICAL_INPUT_LENGTH);
      try {
        HASHER_UPDATE.invokeExact(hasher, input.asSlice(start, sliceLength), sliceLength);
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    }
  }

  @Override
  public void engineUpdate(byte b) {
    oneByteArray[0] = b;
    engineUpdate(oneByteArray, 0, 1);
  }

  @Override
  public void engineUpdate(ByteBuffer input) {
    super.engineUpdate(input);
  }

  private byte[] getOutput(int outputLength) {
    byte[] retByteArray = new byte[outputLength];
    try {
      HASHER_FINALIZE.invokeExact(
          hasher, MemorySegment.ofArray(retByteArray), (long) outputLength);
    } catch (Throwable t) {
      throw new AssertionError(t);
    }

    engineReset();
    return retByteArray;
  }

  /**
   * Frees the native hasher state. Any later use of this instance fails with {@link
   * IllegalStateException}.
   */
  void close() {
    arena.close();
  }

  @Override
  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  @Override
  public void engineReset() {
    try {
      HASHER_INIT.invokeExact(hasher);
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  @Override
  public int engineGetDigestLength() {
    return OUT_LEN;
  }

  @Override
  public byte[] engineDigest() {
    return getOutput(OUT_LEN);
  }

  @Override
  public int engineDigest(byte[] buf, int off, int len) throws DigestException {
    if (len < OUT_LEN) {
      throw new DigestException("partial digests not returned");
    }
    if (buf.length - off < OUT_LEN) {
      throw new DigestException("insufficient space in the output buffer to store the digest");
    }

    byte[] digestBytes = getOutput(OUT_LEN);
    System.arraycopy(digestBytes, 0, buf, off, digestBytes.length);
    return digestBytes.length;
  }
}
