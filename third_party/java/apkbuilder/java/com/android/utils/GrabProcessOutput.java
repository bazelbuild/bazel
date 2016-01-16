/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.utils;

import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.google.common.io.Closeables;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class GrabProcessOutput {

    public enum Wait {
        /**
         * Doesn't wait for the exec to complete.
         * This still monitors the output but does not wait for the process to finish.
         * In this mode the process return code is unknown and always 0.
         */
        ASYNC,
        /**
         * This waits for the process to finish.
         * In this mode, {@link GrabProcessOutput#grabProcessOutput} returns the
         * error code from the process.
         * In some rare cases and depending on the OS, the process might not have
         * finished dumping data into stdout/stderr.
         * <p/>
         * Use this when you don't particularly care for the output but instead
         * care for the return code of the executed process.
         */
        WAIT_FOR_PROCESS,
        /**
         * This waits for the process to finish <em>and</em> for the stdout/stderr
         * threads to complete.
         * In this mode, {@link GrabProcessOutput#grabProcessOutput} returns the
         * error code from the process.
         * <p/>
         * Use this one when capturing all the output from the process is important.
         */
        WAIT_FOR_READERS,
    }

    public interface IProcessOutput {
        /**
         * Processes an stdout message line.
         * @param line The stdout message line. Null when the reader reached the end of stdout.
         */
        public void out(@Nullable String line);
        /**
         * Processes an stderr message line.
         * @param line The stderr message line. Null when the reader reached the end of stderr.
         */
        public void err(@Nullable String line);
    }

    /**
     * Get the stderr/stdout outputs of a process and return when the process is done.
     * Both <b>must</b> be read or the process will block on windows.
     *
     * @param process The process to get the output from.
     * @param output Optional object to capture stdout/stderr.
     *      Note that on Windows capturing the output is not optional. If output is null
     *      the stdout/stderr will be captured and discarded.
     * @param waitMode Whether to wait for the process and/or the readers to finish.
     * @return the process return code.
     * @throws InterruptedException if {@link Process#waitFor()} was interrupted.
     */
    public static int grabProcessOutput(
            @NonNull final Process process,
            Wait waitMode,
            @Nullable final IProcessOutput output) throws InterruptedException {
        // read the lines as they come. if null is returned, it's
        // because the process finished
        Thread threadErr = new Thread("stderr") {
            @Override
            public void run() {
                // create a buffer to read the stderr output
                InputStream is = process.getErrorStream();
                InputStreamReader isr = new InputStreamReader(is);
                BufferedReader errReader = new BufferedReader(isr);

                try {
                    while (true) {
                        String line = errReader.readLine();
                        if (output != null) {
                            output.err(line);
                        }
                        if (line == null) {
                            break;
                        }
                    }
                } catch (IOException e) {
                    // do nothing.
                } finally {
                    try {
                        Closeables.close(is, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                    try {
                        Closeables.close(isr, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                    try {
                        Closeables.close(errReader, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                }
            }
        };

        Thread threadOut = new Thread("stdout") {
            @Override
            public void run() {
                InputStream is = process.getInputStream();
                InputStreamReader isr = new InputStreamReader(is);
                BufferedReader outReader = new BufferedReader(isr);

                try {
                    while (true) {
                        String line = outReader.readLine();
                        if (output != null) {
                            output.out(line);
                        }
                        if (line == null) {
                            break;
                        }
                    }
                } catch (IOException e) {
                    // do nothing.
                } finally {
                    try {
                        Closeables.close(is, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                    try {
                        Closeables.close(isr, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                    try {
                        Closeables.close(outReader, true /* swallowIOException */);
                    } catch (IOException e) {
                        // cannot happen
                    }
                }
            }
        };

        threadErr.start();
        threadOut.start();

        if (waitMode == Wait.ASYNC) {
            return 0;
        }

        // it looks like on windows process#waitFor() can return
        // before the thread have filled the arrays, so we wait for both threads and the
        // process itself.
        if (waitMode == Wait.WAIT_FOR_READERS) {
            try {
                threadErr.join();
            } catch (InterruptedException e) {
            }
            try {
                threadOut.join();
            } catch (InterruptedException e) {
            }
        }

        // get the return code from the process
        return process.waitFor();
    }
}
