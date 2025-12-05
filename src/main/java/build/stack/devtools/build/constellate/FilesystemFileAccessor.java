package build.stack.devtools.build.constellate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import net.starlark.java.syntax.ParserInput;

/** Implementation of {@link StarlarkFileAccessor} which uses the real filesystem. */
public class FilesystemFileAccessor implements StarlarkFileAccessor {

  @Override
  public ParserInput inputSource(String filename) throws IOException {
    return ParserInput.fromLatin1(Files.readAllBytes(Paths.get(filename)), filename);
  }

  @Override
  public boolean fileExists(String pathString) {
    return Files.exists(Paths.get(pathString));
  }
}
