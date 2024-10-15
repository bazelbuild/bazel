package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsProvider;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.stream.JsonReader;
import com.sun.nio.file.ExtendedOpenOption;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousByteChannel;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.Channels;
import java.nio.channels.CompletionHandler;
import java.nio.channels.SocketChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * A file system watcher based on {@see https://facebook.github.io/watchman/}.
 *
 * <p>Since Watchman's watch-project command is used to watch the root of the project, file system
 * events can be reused by other tools such as VCS clients, which reduces overhead compared to using
 * a Bazel-specific file system watcher.
 */
public class WatchmanDiffAwareness extends LocalDiffAwareness {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

  private final ImmutableSet<String> ignoredPaths;
  // The watchman socket is kept open for as long as watchfs is enabled.
  @Nullable private WatchmanSocket watchmanSocket;
  @Nullable private List<Object> nextQueryCommand;

  protected WatchmanDiffAwareness(String watchRoot, ImmutableSet<Path> ignoredPaths) {
    super(watchRoot);
    // Watchman's "dirname" expression matches on root-relative paths.
    this.ignoredPaths =
        ignoredPaths.stream()
            .map(Path.of(watchRoot)::relativize)
            .map(Path::toString)
            .collect(toImmutableSet());
  }

  private void init(PathFragment watchmanPath) throws IOException {
    String rawSockname;
    try {
      var watchmanGetSockname =
          new ProcessBuilder(watchmanPath.getPathString(), "get-sockname").start();
      // The output is short enough that this can't deadlock due to the stdout/stderr buffers
      // filling up.
      int exitCode = watchmanGetSockname.waitFor();
      if (exitCode != 0) {
        String stderr = new String(watchmanGetSockname.getErrorStream().readAllBytes(), UTF_8);
        throw new IOException(
            "watchman get-sockname failed with exit code " + exitCode + ": " + stderr);
      }
      try (BufferedReader reader = watchmanGetSockname.inputReader(UTF_8)) {
        rawSockname = GSON.fromJson(reader, JsonObject.class).get("sockname").getAsString();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return;
    }

    watchmanSocket = WatchmanSocket.connect(Path.of(rawSockname));
  }

  @Override
  public View getCurrentView(OptionsProvider options, EventHandler eventHandler)
      throws BrokenDiffAwarenessException {
    Options watchFsOptions = Preconditions.checkNotNull(options.getOptions(Options.class));
    boolean useWatchman = watchFsOptions.watchFS && !watchFsOptions.watchmanPath.isEmpty();
    // See the comments on WatchServiceDiffAwareness#getCurrentView for the handling of the state
    // changes below.
    if (useWatchman && watchmanSocket == null) {
      try {
        init(watchFsOptions.watchmanPath);
      } catch (IOException e) {
        throw new BrokenDiffAwarenessException(
            "Error encountered with watchman: " + e);
      }
    } else if (!useWatchman && watchmanSocket != null) {
      close();
      throw new BrokenDiffAwarenessException("Switched off --watchfs again");
    }
    if (watchmanSocket == null) {
      return EVERYTHING_MODIFIED;
    }
    Set<Path> modifiedAbsolutePaths;
    if (isFirstCall()) {
      try {
        watchProject(watchRootPath, eventHandler);
      } catch (IOException e) {
        close();
        throw new BrokenDiffAwarenessException(
            "Error encountered with watchman: " + e);
      }
      modifiedAbsolutePaths = ImmutableSet.of();
    } else {
      try {
        modifiedAbsolutePaths = collectChangesAndUpdateClockspec(eventHandler);
      } catch (BrokenDiffAwarenessException e) {
        close();
        throw e;
      } catch (IOException e) {
        close();
        throw new BrokenDiffAwarenessException(
            "Error encountered with local file system watcher " + e);
      }
    }
    if (modifiedAbsolutePaths == null) {
      return EVERYTHING_MODIFIED;
    } else {
      return newView(modifiedAbsolutePaths);
    }
  }

  private void watchProject(Path watchRootPath, EventHandler eventHandler)
      throws IOException, BrokenDiffAwarenessException {
    sendCommand(List.of("watch-project", watchRootPath.toString()));

    String responseJson = watchmanSocket.reader.readLine();
    JsonObject response = GSON.fromJson(responseJson, JsonObject.class);
    if (response.has("error")) {
      throw new IOException(response.get("error").getAsString());
    } else if (response.has("warning")) {
      String warning = response.get("warning").getAsString();
      logger.atInfo().log("Watchman warning: %s", warning);
      eventHandler.handle(
          Event.warn("Watchman returned a warning while registering for file changes: " + warning));
    }

    String watchRoot =
        Optional.ofNullable(response.get("watch"))
            .orElseThrow(
                () ->
                    new IOException(
                        "Watchman 'watch-project' response did not have a 'watch' field"))
            .getAsString();
    Optional<String> watchPrefix =
        Optional.ofNullable(response.get("relative_path")).map(JsonElement::getAsString);
    nextQueryCommand = makeInitialQueryCommand(watchRoot, watchPrefix);

    // The first response to the query command will return no files and be marked as a fresh
    // instance, but provides us with the clockspec relative to which can we can send the next
    // query.
    var ignored = collectChangesAndUpdateClockspec(eventHandler);
  }

  private ImmutableSet<Path> collectChangesAndUpdateClockspec(EventHandler eventHandler)
      throws IOException, BrokenDiffAwarenessException {
    sendCommand(nextQueryCommand);

    boolean isFreshInstance = false;
    ImmutableSet.Builder<Path> modifiedPaths = ImmutableSet.builder();
    var json = new JsonReader(watchmanSocket.reader);
    json.beginObject();
    while (json.hasNext()) {
      switch (json.nextName()) {
        case "error" -> throw new BrokenDiffAwarenessException(json.nextString());
        case "warning" -> {
          String warning = json.nextString();
          logger.atInfo().log("Watchman warning: %s", warning);
          eventHandler.handle(
              Event.warn(
                  "Watchman returned a warning while checking for changed files: " + warning));
        }
        case "clock" -> updateClockspec(json.nextString());
        case "is_fresh_instance" -> isFreshInstance = json.nextBoolean();
        case "files" -> {
          json.beginArray();
          while (json.hasNext()) {
            modifiedPaths.add(watchRootPath.resolve(json.nextString()));
          }
          json.endArray();
        }
        default -> json.skipValue();
      }
    }
    json.endObject();

    if (isFreshInstance) {
      return null;
    }
    return modifiedPaths.build();
  }

  private List<Object> makeInitialQueryCommand(String watchRoot, Optional<String> watchPrefix) {
    Map<String, Object> queryParams = new LinkedHashMap<>();
    queryParams.put("fields", List.of("name"));
    // Avoid an unnecessarily long response on the first query by omitting the list of potentially
    // changed (thus at that point, all) files.
    queryParams.put("empty_on_fresh_instance", true);
    watchPrefix.ifPresent(prefix -> queryParams.put("relative_root", prefix));
    // Bazel only needs to know about changed files, not directories.
    // Watchman doesn't follow symlinks and treats junctions as symlinks, so there's no need to
    // filter out the convenience symlinks.
    // https://github.com/facebook/watchman/blob/13d4454123e6321408ba1610e4caa4f49863409f/watchman/fs/FileInformation.cpp#L70
    List<Object> anyof =
        Stream.concat(
                Stream.of("anyof", List.of("type", "d")),
                ignoredPaths.stream().map(ignoredPath -> List.of("dirname", ignoredPath)))
            .toList();
    queryParams.put("expression", List.of("not", anyof));
    return List.of("query", watchRoot, queryParams);
  }

  private void updateClockspec(String clockspec) {
    // The query command has the form ["query", "/foo/bar", {"since": ..., ...}].
    ((Map<String, Object>) nextQueryCommand.get(2)).put("since", clockspec);
  }

  private void sendCommand(List<Object> command) throws IOException {
    GSON.toJson(command, watchmanSocket.writer);
    watchmanSocket.writer.write('\n');
    watchmanSocket.writer.flush();
  }

  @Override
  public void close() {
    if (watchmanSocket != null) {
      try {
        watchmanSocket.close();
      } catch (IOException e) {
        logger.atInfo().withCause(e).log("While closing watchman channel: %s", e.getMessage());
      } finally {
        watchmanSocket = null;
        nextQueryCommand = null;
      }
    }
  }

  /**
   * A persistent connection to Watchman, either via a Unix domain socket or a Windows named pipe.
   */
  private static final class WatchmanSocket implements Closeable {
    private final BufferedReader reader;
    private final BufferedWriter writer;

    private WatchmanSocket(BufferedReader reader, BufferedWriter writer) {
      this.reader = reader;
      this.writer = writer;
    }

    static WatchmanSocket connect(Path path) throws IOException {
      Reader reader;
      Writer writer;
      if (OS.getCurrent() == OS.WINDOWS) {
        // On Windows, Watchman uses a named pipe for communication. This call mimics
        // https://github.com/facebook/watchman/blob/067252e00bb111fc199daf6aeb7e6572613a153b/watchman/java/src/com/facebook/watchman/windowspipe/WindowsNamedPipe.java#L40-L49
        var channel =
            asAsynchronousByteChannel(
                AsynchronousFileChannel.open(
                    path,
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE,
                    ExtendedOpenOption.NOSHARE_READ,
                    ExtendedOpenOption.NOSHARE_WRITE));
        reader = new InputStreamReader(Channels.newInputStream(channel), UTF_8);
        writer = new OutputStreamWriter(Channels.newOutputStream(channel), UTF_8);
      } else {
        var channel = SocketChannel.open(UnixDomainSocketAddress.of(path));
        reader = Channels.newReader(channel, UTF_8);
        writer = Channels.newWriter(channel, UTF_8);
      }
      return new WatchmanSocket(new BufferedReader(reader), new BufferedWriter(writer));
    }

    @Override
    public void close() throws IOException {
      reader.close();
      writer.close();
    }
  }

  /**
   * Returns an {@link AsynchronousByteChannel} that reads from the beginning of the given {@link
   * AsynchronousFileChannel}.
   */
  private static AsynchronousByteChannel asAsynchronousByteChannel(
      AsynchronousFileChannel channel) {
    return new AsynchronousByteChannel() {
      @Override
      public boolean isOpen() {
        return channel.isOpen();
      }

      @Override
      public void close() throws IOException {
        channel.close();
      }

      @Override
      public <A> void read(
          ByteBuffer dst, A attachment, CompletionHandler<Integer, ? super A> handler) {
        channel.read(dst, 0, attachment, handler);
      }

      @Override
      public Future<Integer> read(ByteBuffer dst) {
        return channel.read(dst, 0);
      }

      @Override
      public <A> void write(
          ByteBuffer src, A attachment, CompletionHandler<Integer, ? super A> handler) {
        channel.write(src, 0, attachment, handler);
      }

      @Override
      public Future<Integer> write(ByteBuffer src) {
        return channel.write(src, 0);
      }
    };
  }
}
