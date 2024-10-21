package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.TruthJUnit.assume;
import static com.google.devtools.build.lib.util.StringEncoding.internalToPlatform;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.platformToInternal;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public class StringEncodingTest {

  @Test
  public void testPlatformToInternal(
      @TestParameter({"ascii", "äöüÄÖÜß", "🌱", "羅勒罗勒学名"}) String s) {
    String internal = platformToInternal(s);
    // In the internal encoding, raw bytes are encoded as Latin-1.
    assertThat(StringUnsafe.getInstance().getCoder(internal)).isEqualTo(StringUnsafe.LATIN1);
    String roundtripped = internalToPlatform(internal);
    if (StringUnsafe.getInstance().isAscii(s)) {
      assertThat(roundtripped).isSameInstanceAs(s);
    } else {
      assertThat(roundtripped).isEqualTo(s);
    }
  }

  @Test
  public void testPlatformToInternal_ByteString(
      @TestParameter({"ascii", "äöüÄÖÜß", "🌱", "羅勒罗勒学名"}) String s) {
    ByteString bytes = ByteString.copyFromUtf8(s);
    String internal = platformToInternal(bytes);
    // In the internal encoding, raw bytes are encoded as Latin-1.
    assertThat(StringUnsafe.getInstance().getCoder(internal)).isEqualTo(StringUnsafe.LATIN1);
    String roundtripped =
        internalToPlatform(ByteString.copyFrom(internal, StandardCharsets.ISO_8859_1));
    assertThat(roundtripped).isEqualTo(s);
  }

  @Test
  public void testPlatformToInternal_rawBytesRoundtrip() {
    // Not valid UTF-8
    byte[] rawBytes = new byte[] {0x00, 0x7F, (byte) 0x80, (byte) 0xFE, (byte) 0xFF};
    assertThrows(
        CharacterCodingException.class,
        () ->
            UTF_8
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(rawBytes)));

    // Roundtripping raw bytes through the internal encoding requires Unix and a Latin-1 locale.
    assume().that(OS.getCurrent()).isNotEqualTo(OS.WINDOWS);
    assume()
        .that(Charset.forName(System.getProperty("sun.jnu.encoding")))
        .isEqualTo(StandardCharsets.ISO_8859_1);

    String platform = new String(rawBytes, ISO_8859_1);
    String internal = platformToInternal(platform);
    assertThat(internal).isSameInstanceAs(platform);
    String roundtripped = internalToPlatform(internal);
    assertThat(roundtripped).isSameInstanceAs(internal);
  }

  @Test
  public void testUnicodeToInternal(@TestParameter({"ascii", "äöüÄÖÜß", "🌱", "羅勒罗勒学名"}) String s) {
    String internal = unicodeToInternal(s);
    // In the internal encoding, raw bytes are encoded as Latin-1.
    assertThat(StringUnsafe.getInstance().getCoder(internal)).isEqualTo(StringUnsafe.LATIN1);
    String roundtripped = internalToUnicode(internal);
    if (StringUnsafe.getInstance().isAscii(s)) {
      assertThat(roundtripped).isSameInstanceAs(s);
    } else {
      assertThat(roundtripped).isEqualTo(s);
    }
  }
}
