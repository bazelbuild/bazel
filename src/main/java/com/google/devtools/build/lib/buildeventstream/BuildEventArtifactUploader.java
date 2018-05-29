package com.google.devtools.build.lib.buildeventstream;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport.TransportKind;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import javax.annotation.concurrent.ThreadSafe;

@ThreadSafe
public interface BuildEventArtifactUploader {

  class NullUploader implements BuildEventArtifactUploader {
    @Override
    public PathConverter upload(Set<Path> files) {
      return (Path path) -> pathToUriString(path.getPathString());
    }

    @Override
    public List<TransportKind> supportedTransports() {
      return Arrays.asList(TransportKind.BEP_FILE, TransportKind.BES_GRPC, TransportKind.STUBBY);
    }

    /**
     * Returns the path encoded as an {@link URI}.
     *
     * <p>This concrete implementation returns URIs with "file" as the scheme. For Example: - On Unix
     * the path "/tmp/foo bar.txt" will be encoded as "file:///tmp/foo%20bar.txt". - On Windows the
     * path "C:\Temp\Foo Bar.txt" will be encoded as "file:///C:/Temp/Foo%20Bar.txt"
     *
     * <p>Implementors extending this class for special filesystems will likely need to override this
     * method.
     *
     * @throws URISyntaxException if the URI cannot be constructed.
     */
    @VisibleForTesting
    static String pathToUriString(String path) {
      if (!path.startsWith("/")) {
        // On Windows URI's need to start with a '/'. i.e. C:\Foo\Bar would be file:///C:/Foo/Bar
        path = "/" + path;
      }
      try {
        return new URI(
            "file",
            // Needs to be "" instead of null, so that toString() will append "//" after the
            // scheme.
            // We need this for backwards compatibility reasons as some consumers of the BEP are
            // broken.
            "",
            path,
            null,
            null)
            .toString();
      } catch (URISyntaxException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  PathConverter upload(Set<Path> files) throws IOException, InterruptedException;

  List<TransportKind> supportedTransports();

}
