// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.skyframe.serialization.ByteStringCodec.byteStringCodec;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HexFormat;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.SymbolGenerator;

/**
 * An evaluated .bzl or .scl file (in the form of a {@link Module}) and its associated {@link
 * BzlLoadValue.Key}. Intended to be used as the {@link SymbolGenerator#create owner} of {@link
 * SymbolGenerator} instances created by {@link BzlLoadFunction}.
 *
 * <p>This class exists for two reasons: to allow checking that we are evaluating a .bzl or .scl
 * file (and not some other variety of Starlark environment); and for (de)serialization and
 * interning of exported Starlark values.
 *
 * <p>If a {@link BzlLoadThreadOwner} object differs from one produced from the current {@link
 * BzlLoadValue} at the same key, it indicates that the first thread owner is a "zombie module" left
 * over from a previous Bazel invocation, and should not be used.
 */
// Note that we rely on (1) Module having identity semantics, and (2) the Module object being
// owned by BzlLoadValue and therefore being unique for a given key within a single Bazel
// invocation.
public record BzlLoadThreadOwner(BzlLoadValue.Key key, Module module) {

  private static final Interner<BzlLoadThreadOwner> interner = BlazeInterners.newWeakInterner();

  public BzlLoadThreadOwner {
    checkNotNull(key);
    checkNotNull(module);
  }

  public static BzlLoadThreadOwner of(BzlLoadValue.Key key, Module module) {
    return interner.intern(new BzlLoadThreadOwner(key, module));
  }

  public static BzlLoadThreadOwner of(BzlLoadValue.Key key, BzlLoadValue value) {
    return of(key, value.getModule());
  }

  public static SymbolGenerator<BzlLoadThreadOwner> createGenerator(
      BzlLoadValue.Key key, Module module) {
    return SymbolGenerator.create(BzlLoadThreadOwner.of(key, module));
  }

  public static SymbolGenerator<BzlLoadThreadOwner> createGenerator(
      BzlLoadValue.Key key, BzlLoadValue value) {
    return SymbolGenerator.create(BzlLoadThreadOwner.of(key, value));
  }

  @VisibleForSerialization
  public static DeferredObjectCodec<BzlLoadThreadOwner> codec() {
    return Codec.INSTANCE;
  }

  /**
   * Codec for {@link BzlLoadThreadOwner}.
   *
   * <p>We cannot serialize a {@link Module} directly. Instead, we serialize the module's transitive
   * digest and {@link StarlarkSemantics} fingerprint, and then check at deserialization time
   * whether the digest and fingerprint still match the module in the {@link BzlLoadValue} in
   * skyframe. (If they don't match, the deserialized {@link BzlLoadThreadOwner} is a zombie.)
   */
  @Keep
  private static final class Codec extends DeferredObjectCodec<BzlLoadThreadOwner> {
    private static final Codec INSTANCE = new Codec();

    @Override
    public Class<BzlLoadThreadOwner> getEncodedClass() {
      return BzlLoadThreadOwner.class;
    }

    @Override
    public void serialize(
        SerializationContext context, BzlLoadThreadOwner obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.key(), BzlLoadValue.bzlLoadKeyCodec(), codedOut);
      context.serializeLeaf(
          BuildLanguageOptions.stableFingerprint(obj.module().getSemantics()),
          byteStringCodec(),
          codedOut);
      context.serializeLeaf(
          ByteString.copyFrom(BazelModuleContext.of(obj.module()).bzlTransitiveDigest()),
          byteStringCodec(),
          codedOut);
    }

    @Override
    public DeferredValue<BzlLoadThreadOwner> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      BzlLoadValue.Key key = context.deserializeLeaf(codedIn, BzlLoadValue.bzlLoadKeyCodec());
      ByteString starlarkSemanticsFingerprint = context.deserializeLeaf(codedIn, byteStringCodec());
      byte[] transitiveDigest = context.deserializeLeaf(codedIn, byteStringCodec()).toByteArray();
      var builder = new DeserializationBuilder(key, starlarkSemanticsFingerprint, transitiveDigest);
      context.getSkyValue(key, builder, DeserializationBuilder::setBzlLoadValue);
      return builder;
    }
  }

  private static ByteString getSemanticsFingerprint(Module module) {
    return BuildLanguageOptions.stableFingerprint(module.getSemantics());
  }

  private static byte[] getTransitiveDigest(Module module) {
    return BazelModuleContext.of(module).bzlTransitiveDigest();
  }

  private static final class DeserializationBuilder implements DeferredValue<BzlLoadThreadOwner> {
    private final BzlLoadValue.Key key;
    private final ByteString starlarkSemanticsFingerprint;
    private final byte[] transitiveDigest;
    private BzlLoadValue loadValue;

    private static final HexFormat HEX_FORMAT = HexFormat.of().withLowerCase();

    private DeserializationBuilder(
        BzlLoadValue.Key key, ByteString starlarkSemanticsFingerprint, byte[] transitiveDigest) {
      this.key = key;
      this.starlarkSemanticsFingerprint = starlarkSemanticsFingerprint;
      this.transitiveDigest = transitiveDigest;
    }

    @Override
    public BzlLoadThreadOwner call() throws SerializationException {
      if (loadValue == null) {
        throw new SerializationException(
            String.format(
                "Failed to retrieve BzlLoadValue for %s; either Skyframe lookup value is not set,"
                    + " or we are attempting to retrieve a zombie module.",
                key));
      }
      Module module = loadValue.getModule();
      if (!(starlarkSemanticsFingerprint.equals(getSemanticsFingerprint(module))
          && Arrays.equals(transitiveDigest, getTransitiveDigest(module)))) {
        throw new SerializationException(
            String.format(
                "Cannot retrieve a zombie module. Expected semantics fingerprint %s and transitive"
                    + " digest %s, but got semantics fingerprint %s and transitive digest %s",
                starlarkSemanticsFingerprint,
                HEX_FORMAT.formatHex(transitiveDigest),
                getSemanticsFingerprint(module),
                HEX_FORMAT.formatHex(getTransitiveDigest(module))));
      }
      return BzlLoadThreadOwner.of(key, module);
    }

    private static void setBzlLoadValue(DeserializationBuilder builder, Object value) {
      builder.loadValue = (BzlLoadValue) value;
    }
  }
}
