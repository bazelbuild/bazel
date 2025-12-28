package build.stack.devtools.build.constellate;

import java.io.IOException;
import net.starlark.java.syntax.ParserInput;

/**
 * A {@link StarlarkFileAccessor} that provides in-memory content for a specific file
 * while delegating to a filesystem accessor for all other files.
 *
 * <p>This is useful for LSP scenarios where the target file content comes from an
 * unsaved editor buffer, but dependencies still need to be loaded from disk.
 */
public class HybridFileAccessor implements StarlarkFileAccessor {

  private final String virtualFilePath;
  private final String virtualFileContent;
  private final StarlarkFileAccessor delegate;

  /**
   * Creates a hybrid file accessor.
   *
   * @param virtualFilePath the absolute path of the file with in-memory content
   * @param virtualFileContent the in-memory content for the virtual file
   * @param delegate the file accessor to use for all other files
   */
  public HybridFileAccessor(
      String virtualFilePath,
      String virtualFileContent,
      StarlarkFileAccessor delegate) {
    this.virtualFilePath = virtualFilePath;
    this.virtualFileContent = virtualFileContent;
    this.delegate = delegate;
  }

  @Override
  public ParserInput inputSource(String pathString) throws IOException {
    if (matchesVirtualFile(pathString)) {
      // Return in-memory content for the virtual file
      return ParserInput.fromUTF8(virtualFileContent.getBytes(), pathString);
    }
    // Delegate to filesystem for all other files
    return delegate.inputSource(pathString);
  }

  @Override
  public boolean fileExists(String pathString) {
    if (matchesVirtualFile(pathString)) {
      // Virtual file always "exists"
      return true;
    }
    // Delegate to filesystem for all other files
    return delegate.fileExists(pathString);
  }

  /**
   * Checks if the given path matches the virtual file.
   * Handles paths with or without depRoot prefixes (e.g., "virtual/inline.bzl" or "./virtual/inline.bzl").
   */
  private boolean matchesVirtualFile(String pathString) {
    // Exact match
    if (pathString.equals(virtualFilePath)) {
      return true;
    }
    // Match with depRoot prefix (e.g., "./virtual/file.bzl" matches "virtual/file.bzl")
    // Check if path ends with the virtual file path
    if (pathString.endsWith("/" + virtualFilePath) || pathString.endsWith(virtualFilePath)) {
      return true;
    }
    return false;
  }
}
