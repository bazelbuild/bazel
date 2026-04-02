package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum.InvalidChecksumException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ChecksumTest {

  // echo -n "sha256-"; openssl dgst -sha256 -binary /dev/null | openssl base64 -A; echo
  private static final String SHA256_INTEGRITY_EMPTY =
      "sha256-O4v8fJvidQTu5H/np7Vl1oly/FDDfzthaP8khIm9PM0=";

  // echo "bazel" > bazel.txt
  // echo -n "sha256-"; openssl dgst -sha256 -binary bazel.txt | openssl base64 -A; echo
  private static final String SHA256_INTEGRITY_BAZEL = "sha256-DfROnrnkTO+tVhos1X9jVCC7+9oG95BU9YmZLHyREac=";

  // echo -n "sha384-"; openssl dgst -sha384 -binary /dev/null | openssl base64 -A; echo
  private static final String SHA384_INTEGRITY_EMPTY =
      "sha384-OLBgp1GsljhM2TJ+sbHjaiH9txEUvgdDTAzHv2P24donTt6/529l+9Ua0vFImLlb";

  // echo -n "sha512-"; openssl dgst -sha512 -binary /dev/null | openssl base64 -A; echo
  private static final String SHA512_INTEGRITY_EMPTY =
      "sha512-z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXcg/SpIdNs6c5H0NE8XYXysP+DGNKHfuwvY7kxvUdBeoGlODJ6+SfaPg==";

  @Test
  public void fromSubresourceIntegrity_valid() throws InvalidChecksumException {
    Checksum checksum256 = Checksum.fromSubresourceIntegrity(SHA256_INTEGRITY_EMPTY);
    assertThat(checksum256.getKeyType()).isEqualTo(KeyType.SHA256);
    String sha256Base64 = "sha256-" + Base64.getEncoder().encodeToString(checksum256.getHashCode().asBytes());
    assertThat(SHA256_INTEGRITY_EMPTY).isEqualTo(sha256Base64);

    Checksum checksum384 = Checksum.fromSubresourceIntegrity(SHA384_INTEGRITY_EMPTY);
    assertThat(checksum384.getKeyType()).isEqualTo(KeyType.SHA384);
    String sha384Base64 = "sha384-" + Base64.getEncoder().encodeToString(checksum384.getHashCode().asBytes());
    assertThat(SHA384_INTEGRITY_EMPTY).isEqualTo(sha384Base64);

    Checksum checksum512 = Checksum.fromSubresourceIntegrity(SHA512_INTEGRITY_EMPTY);
    assertThat(checksum512.getKeyType()).isEqualTo(KeyType.SHA512);
    String sha512Base64 = "sha512-" + Base64.getEncoder().encodeToString(checksum512.getHashCode().asBytes());
    assertThat(SHA512_INTEGRITY_EMPTY).isEqualTo(sha512Base64);
  }

  @Test
  public void fromSubresourceIntegrity_strongestAlgorithm() throws InvalidChecksumException {
    Checksum checksum =
        Checksum.fromSubresourceIntegrity(SHA512_INTEGRITY_EMPTY + "    " + SHA384_INTEGRITY_EMPTY);
    assertThat(checksum.getKeyType()).isEqualTo(KeyType.SHA512);

    assertThat(
            Checksum.fromSubresourceIntegrity(
                    SHA256_INTEGRITY_EMPTY + " " + SHA384_INTEGRITY_EMPTY)
                .getKeyType())
        .isEqualTo(KeyType.SHA384);
  }

  @Test
  public void fromSubresourceIntegrity_ignoresInvalidUnknownAlgorithmsHashes()
      throws InvalidChecksumException {
    String integrityAttr =
        String.join(
            " ",
            List.of(
                "randomstring",
                "ed25519-JrQLj5P/89iXES9+vFgrIy29clF9CC/oPPsw3c5D0bs=",
                "sha256-tooshort",
                "sha256-invalid_characters",
                SHA256_INTEGRITY_EMPTY + "?foo=bar",
                SHA384_INTEGRITY_EMPTY));

    List<String> parseWarnings = new ArrayList<>();
    Checksum checksum = Checksum.fromSubresourceIntegrity(integrityAttr, parseWarnings);

    assertThat(checksum.getKeyType()).isEqualTo(KeyType.SHA384);

    String sha384Base64 = "sha384-" + Base64.getEncoder().encodeToString(checksum.getHashCode().asBytes());
    assertThat(SHA384_INTEGRITY_EMPTY).isEqualTo(sha384Base64);

    String allWarnings = String.join(";", parseWarnings);
    assertThat(allWarnings).contains("Unknown hash algorithm for integrity: 'randomstring'.");
    assertThat(allWarnings).contains("Unknown hash algorithm for integrity: 'ed25519-JrQLj5P/89iXES9+vFgrIy29clF9CC/oPPsw3c5D0bs='.");
    assertThat(allWarnings).contains("Ignoring invalid checksum 'sha256-tooshort'.");
    assertThat(allWarnings).contains(String.format("Ignoring unknown integrity options '%s' from integrity '%s'.", "foo=bar", SHA256_INTEGRITY_EMPTY + "?foo=bar"));
    assertThat(allWarnings).contains("Ignoring invalid base64 'sha256-invalid_characters'.");
  }

  @Test
  public void fromSubresourceIntegrity_duplicateAlgorithmThrows() {
    String integrityAttr =
        String.join(
            " ",
            List.of(
                SHA256_INTEGRITY_EMPTY,
                SHA256_INTEGRITY_BAZEL));

    InvalidChecksumException e = assertThrows(InvalidChecksumException.class,
        () -> {
          Checksum.fromSubresourceIntegrity(integrityAttr);
        });
    assertThat(e).hasMessageThat().contains("Duplicate hash algorithm in list of integrity hashes");
    assertThat(e).hasMessageThat().contains(SHA256_INTEGRITY_EMPTY);
    assertThat(e).hasMessageThat().contains(SHA256_INTEGRITY_BAZEL);
  }

  @Test
  public void fromSubresourceIntegrity_noValidAlgorithmThrows() {
       InvalidChecksumException e = assertThrows(InvalidChecksumException.class,
        () -> {
          Checksum.fromSubresourceIntegrity("foobar bad checksums");
        });
    assertThat(e).hasMessageThat().contains("No valid checksums found in integrity 'foobar bad checksums'");
  }
}
