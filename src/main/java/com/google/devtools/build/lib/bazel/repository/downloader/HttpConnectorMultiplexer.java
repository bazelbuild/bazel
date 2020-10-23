// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.bazel.repository.downloader.RetryingInputStream.Reconnector;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Sleeper;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Class for establishing HTTP connections.
 *
 * <p>This is the most amazing way to download files ever. It makes Bazel builds as reliable as
 * Blaze builds in Google's internal hermettically sealed repository. But this class isn't just
 * reliable. It's also fast. It even works on the worst Internet connections in the farthest corners
 * of the Earth. You are just not going to believe how fast and reliable this design is. It's
 * incredible. Your builds are never going to break again due to downloads. You're going to be so
 * happy. Your developer community is going to be happy. Mr. Jenkins will be happy too. Everyone is
 * going to have such a magnificent developer experience due to the product excellence of this
 * class.
 */
@ThreadSafe
final class HttpConnectorMultiplexer {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final int MAX_THREADS_PER_CONNECT = 2;
  private static final long FAILOVER_DELAY_MS = 2000;
  private static final ImmutableMap<String, String> REQUEST_HEADERS =
      ImmutableMap.of(
          "Accept-Encoding",
          "gzip",
          "User-Agent",
          "Bazel/" + BlazeVersionInfo.instance().getReleaseName());

  private final EventHandler eventHandler;
  private final HttpConnector connector;
  private final HttpStream.Factory httpStreamFactory;
  private final Clock clock;
  private final Sleeper sleeper;

  /**
   * Creates a new instance.
   *
   * <p>Instances are thread safe and can be reused.
   */
  HttpConnectorMultiplexer(
      EventHandler eventHandler,
      HttpConnector connector,
      HttpStream.Factory httpStreamFactory,
      Clock clock,
      Sleeper sleeper) {
    this.eventHandler = eventHandler;
    this.connector = connector;
    this.httpStreamFactory = httpStreamFactory;
    this.clock = clock;
    this.sleeper = sleeper;
  }

  public HttpStream connect(List<URL> urls, Optional<Checksum> checksum) throws IOException {
    return connect(
        urls, checksum, ImmutableMap.<URI, Map<String, String>>of(), Optional.<String>absent());
  }

  public HttpStream connect(
      List<URL> urls, Optional<Checksum> checksum, Map<URI, Map<String, String>> authHeaders)
      throws IOException {
    return connect(urls, checksum, authHeaders, Optional.<String>absent());
  }

  /**
   * Establishes reliable HTTP connection to a good mirror URL.
   *
   * <p>This routine supports HTTP redirects in an RFC compliant manner. It requests gzip content
   * encoding when appropriate in order to minimize bandwidth consumption when downloading
   * uncompressed files. It reports download progress. It enforces a SHA-256 checksum which
   * continues to be enforced even after this method returns.
   *
   * <p>This routine spawns {@value #MAX_THREADS_PER_CONNECT} threads that initiate connections in
   * parallel to {@code urls} with a {@value #FAILOVER_DELAY_MS} millisecond failover waterfall so
   * earlier mirrors are preferred. Each connector thread retries automatically on transient errors
   * with exponential backoff. It vets the first 32kB of any payload before selecting a mirror in
   * order to evade captive portals and avoid ultra-low-bandwidth servers. Even after this method
   * returns the reliability doesn't stop. Each read operation will intercept timeouts and errors
   * and block until the connection can be renegotiated transparently right where it left off.
   *
   * @param urls mirrors by preference; each URL can be: file, http, or https
   * @param checksum checksum lazily checked on entire payload, or empty to disable
   * @return an {@link InputStream} of response payload
   * @param type extension, e.g. "tar.gz" to force on downloaded filename, or empty to not do this
   * @throws IOException if all mirrors are down and contains suppressed exception of each attempt
   * @throws InterruptedIOException if current thread is being cast into oblivion
   * @throws IllegalArgumentException if {@code urls} is empty or has an unsupported protocol
   */
  public HttpStream connect(
      List<URL> urls,
      Optional<Checksum> checksum,
      Map<URI, Map<String, String>> authHeaders,
      Optional<String> type)
      throws IOException {
    HttpUtils.checkUrlsArgument(urls);
    if (Thread.interrupted()) {
      throw new InterruptedIOException();
    }
    // If there's only one URL then there's no need for us to run all our fancy thread stuff.
    if (urls.size() == 1) {
      return establishConnection(urls.get(0), checksum, authHeaders, type);
    }
    MutexConditionSharedMemory context = new MutexConditionSharedMemory();
    // The parent thread always holds the lock except when released by wait().
    synchronized (context) {
      // Create the jobs for workers to do.
      long now = clock.currentTimeMillis();
      long startAtTime = now;
      for (URL url : urls) {
        context.jobs.add(new WorkItem(url, checksum, startAtTime, authHeaders));
        startAtTime += FAILOVER_DELAY_MS;
      }
      // Create the worker thread pool.
      for (int i = 0; i < Math.min(urls.size(), MAX_THREADS_PER_CONNECT); i++) {
        Thread thread = new Thread(new Worker(context), "HttpConnector");
        // These threads will not start doing anything until we release the lock below.
        thread.start();
        context.threads.add(thread);
      }
      // Wait for the first worker to compute a result, or for all workers to fail.
      boolean interrupted = false;
      while (context.result == null && !context.threads.isEmpty()) {
        try {
          // Please note that waiting on a conndition releases the mutex. It also throws
          // InterruptedException if the thread is *already* interrupted.
          context.wait();
        } catch (InterruptedException e) {
          // The interrupted state of this thread is now cleared, so we can call wait() again.
          interrupted = true;
          // We need to terminate the workers before rethrowing InterruptedException.
          break;
        }
      }
      // Now that we have the answer or are interrupted, we need to terminate any remaining workers.
      for (Thread thread : context.threads) {
        thread.interrupt();
      }
      // Now wait for all threads to exit. We technically don't need to do this, but it helps with
      // the regression testing of this implementation.
      while (!context.threads.isEmpty()) {
        try {
          context.wait();
        } catch (InterruptedException e) {
          // We don't care right now. Leave us alone.
          interrupted = true;
        }
      }
      // Now that the workers are terminated, we can safely propagate interruptions.
      if (interrupted) {
        throw new InterruptedIOException();
      }
      // Please do not modify this code to call join() because the way we've implemented this
      // routine is much better and faster. join() is basically a sleep loop when multiple threads
      // exist. By sharing our mutex condition across threads, we were able to make things go
      // lightning fast. If the child threads have not terminated by now, they are guaranteed to do
      // so very soon.
      if (context.result != null) {
        return context.result;
      } else {
        IOException error =
            new IOException("All mirrors are down: " + describeErrors(context.errors));
        // By this point, we probably have a very complex tree of exceptions. Beware!
        for (Throwable workerError : context.errors) {
          error.addSuppressed(workerError);
        }
        throw error;
      }
    }
  }

  private static class MutexConditionSharedMemory {
    @GuardedBy("this") @Nullable HttpStream result;
    @GuardedBy("this") final List<Thread> threads = new ArrayList<>();
    @GuardedBy("this") final Deque<WorkItem> jobs = new LinkedList<>();
    @GuardedBy("this") final List<Throwable> errors = new ArrayList<>();
  }

  private static class WorkItem {
    final URL url;
    final Optional<Checksum> checksum;
    final long startAtTime;
    final Map<URI, Map<String, String>> authHeaders;

    WorkItem(
        URL url,
        Optional<Checksum> checksum,
        long startAtTime,
        Map<URI, Map<String, String>> authHeaders) {
      this.url = url;
      this.checksum = checksum;
      this.startAtTime = startAtTime;
      this.authHeaders = authHeaders;
    }
  }

  private class Worker implements Runnable {
    private final MutexConditionSharedMemory context;

    Worker(MutexConditionSharedMemory context) {
      this.context = context;
    }

    @Override
    public void run() {
      while (true) {
        WorkItem work;
        synchronized (context) {
          // A lot could have happened while we were waiting for this lock. Let's check.
          if (context.result != null
              || context.jobs.isEmpty()
              || Thread.currentThread().isInterrupted()) {
            tellParentThreadWeAreDone();
            return;
          }
          // Now remove a the first job from the fifo.
          work = context.jobs.pop();
        }
        // Wait if necessary before starting this thread.
        long now = clock.currentTimeMillis();
        // Java does not have a true monotonic clock; but since currentTimeMillis returns UTC, it's
        // monotonic enough for our purposes. This routine will not be pwnd by DST or JVM freezes.
        // However it may be trivially impacted by system clock skew correction that go backwards.
        if (now < work.startAtTime) {
          try {
            sleeper.sleepMillis(work.startAtTime - now);
          } catch (InterruptedException e) {
            // The parent thread or JVM has asked us to terminate this thread.
            synchronized (context) {
              tellParentThreadWeAreDone();
              return;
            }
          }
        }
        // Now we're actually going to attempt to connect to the remote server.
        HttpStream result;
        try {
          result =
              establishConnection(
                  work.url, work.checksum, work.authHeaders, Optional.<String>absent());
        } catch (SocketTimeoutException e) {
          // SocketTimeoutException derives from InterruptedIOException, but its occurrence
          // is truly exceptional, so we handle it separately here. Failing to do so hides
          // our exception from the user so that they only see an inscrutable "thread
          // interrupted" message instead.
          synchronized (context) {
            context.errors.add(e);
            continue;
          }
        } catch (InterruptedIOException e) {
          // The parent thread got its result from another thread and killed this one.
          synchronized (context) {
            tellParentThreadWeAreDone();
            return;
          }
        } catch (Exception e) {
          // Oh no the connector failed for some reason. We won't let that interfere with our plans.
          synchronized (context) {
            context.errors.add(e);
            continue;
          }
        }
        // Our connection attempt succeeded! Let's inform the parent thread of this joyous occasion.
        synchronized (context) {
          if (context.result == null) {
            context.result = result;
            result = null;
          }
          tellParentThreadWeAreDone();
        }
        // We created a connection but we lost the race. Now we need to close it outside the mutex.
        // We're not going to slow the parent thread down waiting for this operation to complete.
        if (result != null) {
          try {
            result.close();
          } catch (IOException | RuntimeException e) {
            logger.atWarning().withCause(e).log("close() failed in loser zombie thread");
          }
        }
      }
    }

    @GuardedBy("context")
    private void tellParentThreadWeAreDone() {
      // Remove this thread from the list of threads so parent thread knows when all have exited.
      context.threads.remove(Thread.currentThread());
      // Wake up parent thread so it can check if that list is empty.
      context.notify();
    }
  }

  public static Function<URL, ImmutableMap<String, String>> getHeaderFunction(
      Map<String, String> baseHeaders, Map<URI, Map<String, String>> additionalHeaders) {
    return new Function<URL, ImmutableMap<String, String>>() {
      @Override
      public ImmutableMap<String, String> apply(URL url) {
        ImmutableMap<String, String> headers = ImmutableMap.copyOf(baseHeaders);
        try {
          if (additionalHeaders.containsKey(url.toURI())) {
            Map<String, String> newHeaders = new HashMap<>(headers);
            newHeaders.putAll(additionalHeaders.get(url.toURI()));
            headers = ImmutableMap.copyOf(newHeaders);
          }
        } catch (URISyntaxException e) {
          // If we can't convert the URL to a URI (because it is syntactically malformed), still
          // try to
          // do the connection, not adding authentication information as we cannot look it up.
        }
        return headers;
      }
    };
  }

  private HttpStream establishConnection(
      final URL url,
      Optional<Checksum> checksum,
      Map<URI, Map<String, String>> additionalHeaders,
      Optional<String> type)
      throws IOException {
    final Function<URL, ImmutableMap<String, String>> headerFunction =
        getHeaderFunction(REQUEST_HEADERS, additionalHeaders);
    final URLConnection connection = connector.connect(url, headerFunction);
    return httpStreamFactory.create(
        connection,
        url,
        checksum,
        new Reconnector() {
          @Override
          public URLConnection connect(Throwable cause, ImmutableMap<String, String> extraHeaders)
              throws IOException {
            eventHandler.handle(
                Event.progress(String.format("Lost connection for %s due to %s", url, cause)));
            return connector.connect(
                connection.getURL(),
                new Function<URL, ImmutableMap<String, String>>() {
                  @Override
                  public ImmutableMap<String, String> apply(URL url) {
                    return new ImmutableMap.Builder<String, String>()
                        .putAll(headerFunction.apply(url))
                        .putAll(extraHeaders)
                        .build();
                  }
                });
          }
        },
        type);
  }

  private static String describeErrors(Collection<Throwable> errors) {
    return errors
        .stream()
        .map(Throwable::getMessage)
        .filter(Predicates.notNull())
        .collect(toImmutableSortedSet(Ordering.natural()))
        .toString();
  }
}
