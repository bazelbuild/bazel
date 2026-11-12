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

import com.google.devtools.build.lib.jni.JniLoader;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.security.DigestException;
import java.security.MessageDigest;
import java.util.Objects;

/** A {@link MessageDigest} for BLAKE3. */
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
  // possible to pass heap arrays to the native functions directly, without
  // copying. A critical call delays safepoints until it returns, so each call
  // hashes at most MAX_CRITICAL_INPUT_LENGTH bytes.
  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup LOOKUP = SymbolLookup.loaderLookup();
  private static final MemoryLayout SIZE_T = LINKER.canonicalLayouts().get("size_t");
  private static final MethodHandle HASHER_UPDATE =
      LINKER.downcallHandle(
          findSymbol("blake3_hasher_update"),
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, SIZE_T),
          Linker.Option.critical(/* allowHeapAccess= */ true));
  private static final MethodHandle HASHER_FINALIZE =
      LINKER.downcallHandle(
          findSymbol("blake3_hasher_finalize"),
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, SIZE_T),
          Linker.Option.critical(/* allowHeapAccess= */ true));
  private static final MethodHandle HASH =
      LINKER.downcallHandle(
          findSymbol("blaze_blake3_hash"),
          FunctionDescriptor.ofVoid(ADDRESS, SIZE_T, ADDRESS),
          Linker.Option.critical(/* allowHeapAccess= */ true));

  // Hashing this many bytes takes tens of microseconds. It is a multiple of the
  // input size that the widest SIMD implementations process in parallel.
  private static final int MAX_CRITICAL_INPUT_LENGTH = 64 * 1024;

  /**
   * Looks up a native symbol, throwing {@link UnsatisfiedLinkError} just like a JNI method would if
   * it is missing. {@link BazelHashFunctions} relies on this error to detect that the native
   * library was built without the blake3 symbols when bootstrapping Bazel.
   */
  private static MemorySegment findSymbol(String name) {
    return LOOKUP
        .find(name)
        .orElseThrow(() -> new UnsatisfiedLinkError("undefined symbol: " + name));
  }

  // A blake3_hasher right after blake3_hasher_init. The struct only consists
  // of integers and arrays, so copies of it are valid hasher states. It
  // contains uint64_t fields, which the elements of a long[] are aligned for.
  private static final long[] INITIAL_STATE;

  static {
    MethodHandle hasherSize =
        LINKER.downcallHandle(
            findSymbol("blaze_blake3_hasher_size"),
            FunctionDescriptor.of(JAVA_INT),
            Linker.Option.critical(/* allowHeapAccess= */ false));
    MethodHandle hasherInit =
        LINKER.downcallHandle(
            findSymbol("blake3_hasher_init"),
            FunctionDescriptor.ofVoid(ADDRESS),
            Linker.Option.critical(/* allowHeapAccess= */ true));
    try {
      int stateSize = (int) hasherSize.invokeExact();
      INITIAL_STATE = new long[(stateSize + Long.BYTES - 1) / Long.BYTES];
      hasherInit.invokeExact(MemorySegment.ofArray(INITIAL_STATE));
    } catch (Throwable t) {
      throw new ExceptionInInitializerError(t);
    }
  }

  private final long[] state = INITIAL_STATE.clone();
  private final byte[] oneByteArray = new byte[1];

  public Blake3MessageDigest() {
    super("BLAKE3");
  }

  /** Returns the BLAKE3 digest of the given range of bytes. */
  static byte[] hash(byte[] data, int offset, int length) {
    var input = MemorySegment.ofArray(data).asSlice(offset, length);
    var out = new byte[OUT_LEN];
    if (length <= MAX_CRITICAL_INPUT_LENGTH) {
      try {
        HASH.invokeExact(input, (long) length, MemorySegment.ofArray(out));
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    } else {
      var state = MemorySegment.ofArray(INITIAL_STATE.clone());
      update(state, input);
      writeOutput(state, out);
    }
    return out;
  }

  private static void update(MemorySegment state, MemorySegment input) {
    long length = input.byteSize();
    for (long start = 0; start < length; start += MAX_CRITICAL_INPUT_LENGTH) {
      long sliceLength = Math.min(length - start, MAX_CRITICAL_INPUT_LENGTH);
      try {
        HASHER_UPDATE.invokeExact(state, input.asSlice(start, sliceLength), sliceLength);
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    }
  }

  private static void writeOutput(MemorySegment state, byte[] out) {
    try {
      HASHER_FINALIZE.invokeExact(state, MemorySegment.ofArray(out), (long) out.length);
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  @Override
  public void engineUpdate(byte[] data, int offset, int length) {
    Objects.checkFromIndexSize(offset, length, data.length);
    update(MemorySegment.ofArray(state), MemorySegment.ofArray(data).asSlice(offset, length));
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

  /** Returns the digest of the input so far without resetting this instance. */
  byte[] digestWithoutReset() {
    var out = new byte[OUT_LEN];
    writeOutput(MemorySegment.ofArray(state), out);
    return out;
  }

  private byte[] getOutput() {
    byte[] out = digestWithoutReset();
    engineReset();
    return out;
  }

  @Override
  public Object clone() throws CloneNotSupportedException {
    throw new CloneNotSupportedException();
  }

  @Override
  public void engineReset() {
    System.arraycopy(INITIAL_STATE, 0, state, 0, state.length);
  }

  @Override
  public int engineGetDigestLength() {
    return OUT_LEN;
  }

  @Override
  public byte[] engineDigest() {
    return getOutput();
  }

  @Override
  public int engineDigest(byte[] buf, int off, int len) throws DigestException {
    if (len < OUT_LEN) {
      throw new DigestException("partial digests not returned");
    }
    if (buf.length - off < OUT_LEN) {
      throw new DigestException("insufficient space in the output buffer to store the digest");
    }

    byte[] digestBytes = getOutput();
    System.arraycopy(digestBytes, 0, buf, off, digestBytes.length);
    return digestBytes.length;
  }
}
