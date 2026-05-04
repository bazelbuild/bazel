// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkPositionIndexes;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.hash.Funnel;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashFunction;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.hash.PrimitiveSink;
import com.google.common.io.ByteArrayDataOutput;
import com.google.common.io.ByteStreams;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;

/** A {@link HashFunction} for GITSHA1. */
public final class GitSha1HashFunction implements HashFunction {
  public static final HashFunction INSTANCE = new GitSha1HashFunction();
  private static final HashFunction SHA1 = Hashing.sha1();
  private static final byte[] header = {'b', 'l', 'o', 'b', ' '};

  private Hasher newInitializedHasher(int blobSize) {
    Hasher hasher = SHA1.newHasher();
    hasher.putBytes(header);
    hasher.putString(Integer.toString(blobSize), UTF_8);
    hasher.putByte((byte) 0);
    return hasher;
  }

  @Override
  public int bits() {
    return 160;
  }

  @Override
  public Hasher newHasher() {
    return new DelayedGitSha1Hasher();
  }

  @Override
  public Hasher newHasher(int expectedInputSize) {
    checkArgument(
        expectedInputSize >= 0, "expectedInputSize must be >= 0 but was %s", expectedInputSize);
    return newInitializedHasher(expectedInputSize);
  }

  /* The following methods implement the {HashFunction} interface. */

  @Override
  public <T> HashCode hashObject(T instance, Funnel<? super T> funnel) {
    return newHasher().putObject(instance, funnel).hash();
  }

  @Override
  public HashCode hashUnencodedChars(CharSequence input) {
    int len = input.length();
    return newHasher(len * 2).putUnencodedChars(input).hash();
  }

  @Override
  public HashCode hashString(CharSequence input, Charset charset) {
    return newHasher(input.length()).putString(input, charset).hash();
  }

  @Override
  public HashCode hashInt(int input) {
    return newHasher(4).putInt(input).hash();
  }

  @Override
  public HashCode hashLong(long input) {
    return newHasher(8).putLong(input).hash();
  }

  @Override
  public HashCode hashBytes(byte[] input) {
    return hashBytes(input, 0, input.length);
  }

  @Override
  public HashCode hashBytes(byte[] input, int off, int len) {
    checkPositionIndexes(off, off + len, input.length);
    return newHasher(len).putBytes(input, off, len).hash();
  }

  @Override
  public HashCode hashBytes(ByteBuffer input) {
    return newHasher(input.remaining()).putBytes(input).hash();
  }

  private class DelayedGitSha1Hasher implements Hasher {
    private final ByteArrayOutput output;

    DelayedGitSha1Hasher() {
      output = new ByteArrayOutput();
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putBoolean(boolean b) {
      output.putByte(b ? (byte) 1 : (byte) 0);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putByte(byte b) {
      output.putByte(b);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putBytes(byte[] bytes) {
      output.putBytes(bytes);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putBytes(byte[] bytes, int off, int len) {
      output.putBytes(bytes, off, len);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putBytes(ByteBuffer b) {
      output.putBytes(b.array(), b.arrayOffset() + b.position(), b.remaining());
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putShort(short s) {
      output.putShort(s);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putInt(int i) {
      output.putInt(i);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putLong(long l) {
      output.putLong(l);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public <T> Hasher putObject(T instance, Funnel<? super T> funnel) {
      funnel.funnel(instance, output);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putChar(char c) {
      output.putChar(c);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putDouble(double d) {
      output.putDouble(d);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putFloat(float f) {
      output.putFloat(f);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putUnencodedChars(CharSequence charSequence) {
      for (int i = 0; i < charSequence.length(); i++) {
        output.putChar(charSequence.charAt(i));
      }
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public Hasher putString(CharSequence charSequence, Charset charset) {
      output.putBytes(charSequence.toString().getBytes(charset));
      return this;
    }

    @Override
    public HashCode hash() {
      byte[] body = output.toByteArray();
      return newHasher(body.length).putBytes(body).hash();
    }

    @Override
    public int hashCode() {
      return hash().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof Hasher hasher) {
        return this.hash().equals(hasher.hash());
      }
      return false;
    }
  }

  private static class ByteArrayOutput implements PrimitiveSink {
    private final ByteArrayDataOutput buffer = ByteStreams.newDataOutput();

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putBoolean(boolean b) {
      buffer.writeBoolean(b);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putByte(byte b) {
      buffer.write(b);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putBytes(byte[] bytes) {
      buffer.write(bytes);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putBytes(byte[] bytes, int off, int len) {
      buffer.write(bytes, off, len);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putBytes(ByteBuffer b) {
      buffer.write(b.array(), b.arrayOffset() + b.position(), b.remaining());
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putChar(char c) {
      buffer.writeChar(c);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putDouble(double d) {
      buffer.writeDouble(d);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putFloat(float f) {
      buffer.writeFloat(f);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putInt(int i) {
      buffer.writeInt(i);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putLong(long l) {
      buffer.writeLong(l);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putShort(short s) {
      buffer.writeShort(s);
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putString(CharSequence charSequence, Charset charset) {
      buffer.write(charSequence.toString().getBytes(charset));
      return this;
    }

    @CanIgnoreReturnValue
    @Override
    public ByteArrayOutput putUnencodedChars(CharSequence charSequence) {
      for (int i = 0; i < charSequence.length(); i++) {
        buffer.writeChar(charSequence.charAt(i));
      }
      return this;
    }

    byte[] toByteArray() {
      return buffer.toByteArray();
    }
  }
}
