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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkValue} serialization. */
@RunWith(JUnit4.class)
public class StarlarkValueCodecTest {
  /** Dummy Starlark type. */
  private static class SkyDummy implements StarlarkValue {

    private final String value;

    public SkyDummy(String value) {
      this.value = value;
    }

    @Override
    public void repr(Printer printer, StarlarkSemantics semantics) {
      printer.append("Dummy");
    }
  }

  private enum InjectedError {
    NONE,
    ENCODING,
    DECODING
  }

  /** Codec for SkyDummy, with support for injecting errors. */
  // Does not end in "Codec" so that we can custom-register the one we want.
  private static class SkyDummyCodecBase implements ObjectCodec<SkyDummy> {

    private final InjectedError injectedError;

    public SkyDummyCodecBase(InjectedError injectedError) {
      this.injectedError = injectedError;
    }

    public SkyDummyCodecBase() {
      this(InjectedError.NONE);
    }

    @Override
    public Class<SkyDummy> getEncodedClass() {
      return SkyDummy.class;
    }

    @Override
    public void serialize(SerializationContext context, SkyDummy obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      if (injectedError == InjectedError.ENCODING) {
        throw new SerializationException("Simulated encoding error");
      }
      codedOut.writeStringNoTag(obj.value);
    }

    @Override
    public SkyDummy deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      if (injectedError == InjectedError.DECODING) {
        throw new SerializationException("Simulated decoding error");
      }
      return new SkyDummy(codedIn.readBytes().toStringUtf8());
    }
  }

  @Test
  public void encodeAndDecode() throws Exception {
    SkyDummy original = new SkyDummy("dummy");
    assertThat(RoundTripping.roundTripMemoized(original, new SkyDummyCodecBase()).value)
        .isEqualTo(original.value);
  }

  @Test
  public void codecFailureToEncodeThrowsException() {
    SerializationException expected =
        assertThrows(
            SerializationException.class,
            () ->
                RoundTripping.toBytesMemoized(
                    new SkyDummy("dummy"),
                    ObjectCodecRegistry.newBuilder()
                        .add(new SkyDummyCodecBase(InjectedError.ENCODING))
                        .build()));
    assertThat(expected).hasMessageThat().contains("Simulated encoding error");
  }

  @Test
  public void codecFailureToDecodeThrowsException() throws IOException, SerializationException {
    ObjectCodecRegistry registry =
        AutoRegistry.get().getBuilder().add(new SkyDummyCodecBase(InjectedError.DECODING)).build();
    ByteString bytes = RoundTripping.toBytesMemoized(new SkyDummy("dummy"), registry);
    SerializationException expected =
        assertThrows(
            SerializationException.class, () -> RoundTripping.fromBytesMemoized(bytes, registry));
    assertThat(expected).hasMessageThat().contains("Simulated decoding error");
  }

  @Test
  public void standardInstanceRejectsUnknownType() throws Exception {
    NoCodecException expected =
        assertThrows(
            NoCodecException.class,
            () ->
                RoundTripping.toBytesMemoized(
                    new SkyDummy("dummy"), ObjectCodecRegistry.newBuilder().build()));
    assertThat(expected)
        .hasMessageThat()
        .matches(
            "No default codec available for com.google.devtools.build.lib.skyframe.serialization"
                + ".StarlarkValueCodecTest\\$SkyDummy .*");
  }

  @Test
  public void builtinShortcircuitsCodec() throws Exception {
    SkyDummy builtin = new SkyDummy("a");
    SkyDummy nonBuiltin = new SkyDummy("b");

    ObjectCodecRegistry registry =
        AutoRegistry.get()
            .getBuilder()
            .add(new SkyDummyCodecBase())
            .addReferenceConstant(builtin)
            .build();
    // We explicitly care that the exact object from the builtins lookup table is returned, not an
    // equivalent object.
    assertThat(RoundTripping.roundTripMemoized(builtin, registry)).isSameInstanceAs(builtin);
    assertThat(RoundTripping.roundTripMemoized(nonBuiltin, registry))
        .isNotSameInstanceAs(nonBuiltin);
  }
}
