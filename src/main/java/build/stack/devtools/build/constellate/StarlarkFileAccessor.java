package build.stack.devtools.build.constellate;

import java.io.IOException;
import net.starlark.java.syntax.ParserInput;

/**
 * Helper to handle constellate file I/O. This abstraction is useful for tests
 * which don't involve actual file I/O.
 */
public interface StarlarkFileAccessor {

  /** Returns a {@link ParserInput} for accessing the content of the given absolute path string. */
  ParserInput inputSource(String pathString) throws IOException;

  /** Returns true if a file exists at the current path. */
  boolean fileExists(String pathString);
}
