package com.google.devtools.build.lib.remote.blobstore.http;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

final class Utils {

  static String constructPath(URI uri, String hash, boolean isCas) {
    StringBuilder builder = new StringBuilder();
    builder.append(uri.getPath());
    if (!uri.getPath().endsWith("/")) {
      builder.append("/");
    }
    builder.append(isCas ? "cas/" : "ac/");
    builder.append(hash);
    return builder.toString();
  }

  static URI constructBlobURI(URI baseURI, String hash) throws IOException {
    try {
      return new URI(baseURI.getScheme(),
          null /* don't forward credentials */,
          baseURI.getHost(),
          baseURI.getPort(),
          constructPath(baseURI, hash, true),
          null,
          null);
    } catch (URISyntaxException e) {
      throw new IOException("Could not construct BLOB path.", e);
    }
  }
}
