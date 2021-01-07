// Copyright 2014 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.io.Files;
import java.io.File;
import java.util.List;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** An EvalException indicates an Starlark evaluation error. */
public class EvalException extends Exception {

  // The location optionally specified at construction.
  // TODO(adonovan): doesn't belong; essentially ignored.
  // Replace by making each caller incorporate
  // the file name into the error message if necessary.
  // In the vast majority of cases, it isn't.
  @Nullable private final Location location;

  // The call stack associated with this error.
  // It is initially null, but is set by the interpreter to a non-empty
  // stack when popping a frame. Thus an exception newly created by a
  // built-in function has no stack until it is thrown out of a function call.
  //
  // EvalExceptions are often used to indicate the failure of an operator
  // such as getattr, or some piece of bazel rule validation machinery,
  // without reference to Starlark code. Such exceptions have no stack
  // until they are thrown in the context of a Starlark thread (e.g.
  // by a built-in function, or by the interpreter itself).
  @Nullable private ImmutableList<StarlarkThread.CallStackEntry> callstack;

  /** Constructs an EvalException. Use {@link Starlak#errorf} if you want string formatting. */
  public EvalException(String message) {
    this((Location) null, message);
  }

  /**
   * Constructs an EvalException with a message and optional cause.
   *
   * <p>The cause does not affect the error message, so callers should incorporate {@code
   * cause.getMessage()} into {@code message} if desired, or call {@code EvalException(Throwable)}.
   */
  public EvalException(String message, @Nullable Throwable cause) {
    this((Location) null, message, cause);
  }

  /** Constructs an EvalException using the same message as the cause exception. */
  public EvalException(Throwable cause) {
    this((Location) null, getCauseMessage(cause), cause);
  }

  private static String getCauseMessage(Throwable cause) {
    String msg = cause.getMessage();
    return msg != null ? msg : cause.toString();
  }

  // TODO(adonovan): delete constructors below. Stop using Location.

  /**
   * Constructs an EvalException with a message and optional location (deprecated).
   *
   * <p>Few clients need this constructor, as the Starlark interpreter automatically fill in the
   * locations from the call stack. Use {@link Starlark#errorf} instead, unless the exception needs
   * to appear to originate from a different location.
   */
  // TODO(adonovan): eliminate.
  public EvalException(@Nullable Location location, String message) {
    super(Preconditions.checkNotNull(message));
    this.location = location;
  }

  /**
   * Constructs an EvalException with a message, optional location (deprecated), and optional cause.
   *
   * <p>See notes at {@link #EvalException(Location, String)}. The cause does not affect the error
   * message, so callers should incorporate {@code cause.getMessage()} into {@code message} if
   * desired.
   */
  private EvalException(@Nullable Location location, String message, @Nullable Throwable cause) {
    super(Preconditions.checkNotNull(message), cause);
    this.location = location;
  }

  /** Returns the error message. Does not include location (deprecated), call stack, or cause. */
  @Override
  public final String getMessage() {
    return super.getMessage();
  }

  /**
   * Returns the call stack associated with this error, outermost call first. A newly constructed
   * exception has an empty stack, but an exception that has been thrown out of a Starlark function
   * call has its stack populated automatically.
   */
  public final ImmutableList<StarlarkThread.CallStackEntry> getCallStack() {
    return callstack != null ? callstack : ImmutableList.of();
  }

  /** Returns the error message along with its call stack. May be overridden by subclasses. */
  @Override
  public String toString() {
    return getMessageWithStack();
  }

  /**
   * Returns the error message along with its call stack or location (deprecated), if any.
   * Equivalent to {@code getMessageWithStack(newSourceReader())}.
   */
  public final String getMessageWithStack() {
    return getMessageWithStack(newSourceReader());
  }

  /**
   * Returns the error message along with its call stack or location (deprecated), if any. The
   * source line for each stack frame is obtained from the provided SourceReader.
   */
  public final String getMessageWithStack(SourceReader src) {
    if (callstack != null) {
      return formatCallStack(callstack, getMessage(), src);
    }

    // An exception that has not been thrown out of a Starlark call
    // has no stack. It may have a location (for now). If so, print it.
    if (location != null && !location.equals(Location.BUILTIN)) {
      return location + ": " + getMessage();
    }

    return getMessage();
  }

  /**
   * A SourceReader reads the line of source denoted by a Location to be displayed in a formatted
   * stack trace.
   */
  public interface SourceReader {
    /** Returns a single line of source code (sans newline), or null if unavailable. */
    String readline(Location loc);
  }

  /**
   * Sets the function used to obtain a SourceReader when subsequently formatting a call stack.
   *
   * <p>The default supplier returns SourceReaders that read from the file system, but a
   * security-conscious client may wish to disable this capability or provide an alternative.
   */
  public static synchronized void setSourceReaderSupplier(Supplier<SourceReader> f) {
    sourceReaderSupplier = f;
  }

  /** Returns a new SourceReader. See {@link #setSourceReaderSupplier}. */
  public static synchronized SourceReader newSourceReader() {
    return sourceReaderSupplier.get();
  }

  private static Supplier<SourceReader> sourceReaderSupplier =
      () -> {
        // TODO(adonovan): opt: cache seen files, as the stack often repeats the same files.
        return loc -> {
          try {
            String content = Files.asCharSource(new File(loc.file()), UTF_8).read();
            return Iterables.get(Splitter.on("\n").split(content), loc.line() - 1, null);
          } catch (Throwable unused) {
            // ignore any failure (e.g. security manager rejecting I/O)
          }
          return null;
        };
      };

  /**
   * Formats the given call stack and error message. Provided as a separate function from {@link
   * #getMessageWithStack} so that clients may modify the stack and/or error before formatting it.
   * The source line for each stack frame is obtained from the provided SourceReader.
   */
  public static String formatCallStack(
      List<StarlarkThread.CallStackEntry> callstack, String message, SourceReader src) {
    StringBuilder buf = new StringBuilder();
    int n = callstack.size(); // n > 0
    String prefix = "Error: ";
    // If the topmost frame is a built-in, don't show it.
    // Instead just prefix the name of the built-in onto the error message.
    StarlarkThread.CallStackEntry leaf = callstack.get(n - 1);
    if (leaf.location.equals(Location.BUILTIN)) {
      prefix = "Error in " + leaf.name + ": ";
      n--;
    }
    if (n > 0) {
      buf.append("Traceback (most recent call last):\n");
      for (int i = 0; i < n; i++) {
        StarlarkThread.CallStackEntry fr = callstack.get(i);
        // 'File "file.bzl", line 1, column 2, in fn'
        buf.append(String.format("\tFile \"%s\", ", fr.location.file()));
        if (fr.location.line() != 0) {
          buf.append("line ").append(fr.location.line()).append(", ");
          if (fr.location.column() != 0) {
            buf.append("column ").append(fr.location.column()).append(", ");
          }
        }
        buf.append("in ").append(fr.name).append('\n');

        // source line
        String line = src.readline(fr.location);
        if (line != null) {
          buf.append("\t\t").append(line.trim()).append('\n');
        }
      }
    }
    buf.append(prefix).append(message);
    return buf.toString();
  }

  /**
   * Returns the optional location passed to the constructor.
   *
   * @deprecated Do not use this feature. Instead, record auxiliary (non-stack) locations in the
   *     error message itself, or call a dummy wrapper function to introduce a fake frame into the
   *     call stack.
   */
  @Nullable
  @Deprecated
  public final Location getDeprecatedLocation() {
    return location;
  }

  // Ensures that this exception holds a call stack, taking the current
  // stack (which must be non-empty) from the thread if not.
  final EvalException ensureStack(StarlarkThread thread) {
    if (callstack == null) {
      this.callstack = thread.getCallStack();
      if (callstack.isEmpty()) {
        throw new IllegalStateException("empty callstack");
      }
    }
    return this;
  }
}
