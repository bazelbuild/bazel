package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyFactory;
import java.security.KeyManagementException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;

/**
 * Helper methods to instantiate an {@link SSLContext}.
 */
class SSLContextBuilder {

  /**
   * Instantiates an {@link SSLContext} from user-supplied configuration options.
   *
   * @param authAndTLSOptions the authentication options that specify where to find the certificates
   *                          and keys to use
   * @return the new {@link SSLContext} or null if there are no configured certificates or keys
   * @throws IOException if there is any problem parsing the specified files
   */
  public static SSLContext build(AuthAndTLSOptions authAndTLSOptions) throws IOException {
    String certPath = authAndTLSOptions.tlsClientCertificate;
    String keyPath = authAndTLSOptions.tlsClientKey;

    // Exit early if the user has provided no certificates.
    if (certPath == null || keyPath == null) {
      return null;
    }

    try {
      Certificate[] certificates = parseCertificates(certPath);
      PrivateKey privateKey = parsePrivateKey(keyPath);

      KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
      keyStore.load(null, null);
      keyStore.setKeyEntry("client", privateKey, new char[0], certificates);

      KeyManagerFactory keyManagerFactory = KeyManagerFactory.getInstance(
          KeyManagerFactory.getDefaultAlgorithm());
      keyManagerFactory.init(keyStore, new char[0]);

      SSLContext sslContext = SSLContext.getInstance("TLS");
      sslContext.init(keyManagerFactory.getKeyManagers(), null, null);

      return sslContext;
    } catch (IOException | CertificateException | NoSuchAlgorithmException | KeyStoreException
             | UnrecoverableKeyException | KeyManagementException | InvalidKeySpecException e) {
      throw new IOException("Failed to build SSLContext", e);
    }
  }

  /**
   * Extracts and decodes multiple base64 chunks from a PKCS#12 file.
   *
   * @param path   path to the file to parse
   * @param marker string marker delimiting the various chunks in the file
   * @return a collection of base64-decoded chunks
   * @throws IOException if the content of the file is invalid
   */
  @VisibleForTesting
  static List<byte[]> parseChunks(String path, String marker)
      throws IOException {
    List<byte[]> chunks = new ArrayList<>();

    String beginMarker = String.format("-----BEGIN %s-----", marker);
    String endMarker = String.format("-----END %s-----", marker);

    String content = Files.readString(Paths.get(path));
    String[] lines = content.split("\\n");
    StringBuilder base64 = null;
    for (String line : lines) {
      line = line.trim();

      if (line.equals(beginMarker)) {
        if (base64 != null) {
          throw new IOException("Malformed file: nested BEGIN tags");
        }

        base64 = new StringBuilder();
      } else if (line.equals(endMarker)) {
        if (base64 == null) {
          throw new IOException("Malformed file: END tag without BEGIN");
        }

        byte[] bytes = Base64.getDecoder().decode(base64.toString());
        chunks.add(bytes);

        base64 = null;
      } else {
        if (base64 == null) {
          throw new IOException("Malformed file: content without BEGIN");
        }

        base64.append(line);
      }
    }
    if (base64 != null) {
      throw new IOException("Malformed file: BEGIN tag without END");
    }

    return chunks;
  }

  /**
   * Extracts and decodes multiple certificates from a PKCS#12 file.
   *
   * @param certPath path to the file to parse
   * @return a collection of certificates
   * @throws IOException if the content of the file is invalid
   */
  @VisibleForTesting
  static Certificate[] parseCertificates(String certPath)
      throws IOException, CertificateException {
    List<byte[]> chunks = parseChunks(certPath, "CERTIFICATE");
    Certificate[] certificates = new Certificate[chunks.size()];
    int i = 0;
    for (byte[] chunk : chunks) {
      CertificateFactory certFactory = CertificateFactory.getInstance("X.509");
      Certificate certificate = certFactory.generateCertificate(
          new ByteArrayInputStream(chunk));
      certificates[i++] = certificate;
    }
    return certificates;
  }

  /**
   * Extracts and decodes a single private key from a PKCS#12 file.
   *
   * @param keyPath path to the file to parse
   * @return a private key
   * @throws IOException if the content of the file is invalid
   */
  @VisibleForTesting
  static PrivateKey parsePrivateKey(String keyPath)
      throws IOException, NoSuchAlgorithmException, InvalidKeySpecException {
    List<byte[]> chunks = parseChunks(keyPath, "PRIVATE KEY");
    if (chunks.size() != 1) {
      throw new IOException("Expected only one private key");
    }
    PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(chunks.getFirst());

    // Try different algorithms as we don't know the key type from PKCS#8 format alone.
    String[] algorithms = {"RSA", "EC", "DSA"};
    for (String algorithm : algorithms) {
      try {
        KeyFactory keyFactory = KeyFactory.getInstance(algorithm);
        return keyFactory.generatePrivate(keySpec);
      } catch (InvalidKeySpecException e) {
        // Ignore; assume that the algorithm we tried was inappropriate.
      }
    }

    throw new InvalidKeySpecException("Could not parse private key with any supported algorithm");
  }
}
