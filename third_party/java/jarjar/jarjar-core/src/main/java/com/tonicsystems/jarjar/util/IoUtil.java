/**
 * Copyright 2008 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.tonicsystems.jarjar.util;

import java.io.*;
import javax.annotation.Nonnull;
import javax.annotation.WillNotClose;

public class IoUtil {

    private IoUtil() {
    }

    public static void copy(@Nonnull @WillNotClose InputStream is, @Nonnull @WillNotClose OutputStream out, @Nonnull byte[] buf) throws IOException {
        for (;;) {
            int amt = is.read(buf);
            if (amt < 0)
                break;
            out.write(buf, 0, amt);
        }
    }

    public static void copy(@Nonnull File from, @Nonnull File to, @Nonnull byte[] buf) throws IOException {
        InputStream in = new FileInputStream(from);
        try {
            OutputStream out = new FileOutputStream(to);
            try {
                copy(in, out, buf);
            } finally {
                out.close();
            }
        } finally {
            in.close();
        }
    }

    public static void flush(Object o) throws IOException {
        if (o instanceof Flushable)
            ((Flushable) o).flush();
    }

    public static void close(Object o) throws IOException {
        if (o instanceof Closeable)
            ((Closeable) o).close();
    }

}
