/**
 * Copyright 2007 Google Inc.
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
package com.tonicsystems.jarjar.transform.config;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.WillClose;

public class RulesFileParser {

    public interface Output {

        public void addClassDelete(@Nonnull ClassDelete classDelete);

        public void addClassRename(@Nonnull ClassRename classRename);

        public void addClassKeep(@Nonnull ClassKeep classKeep);

        public void addClassKeepTransitive(@Nonnull ClassKeepTransitive classKeepTransitive);
    }

    private RulesFileParser() {
    }

    @Nonnull
    public static void parse(@Nonnull Output output, @Nonnull File file) throws IOException {
        parse(output, new FileReader(file));
    }

    @Nonnull
    public static void parse(@Nonnull Output output, @Nonnull String value) throws IOException {
        parse(output, new StringReader(value));
    }

    @Nonnull
    private static List<String> split(@Nonnull String line) {
        StringTokenizer tok = new StringTokenizer(line);
        List<String> out = new ArrayList<String>();
        while (tok.hasMoreTokens()) {
            String token = tok.nextToken();
            if (token.startsWith("#"))
                break;
            out.add(token);
        }
        return out;
    }

    @Nonnull
    private static void parse(@Nonnull Output output, @Nonnull @WillClose Reader r) throws IOException {
        try {
            BufferedReader br = new BufferedReader(r);
            int lineNumber = 1;
            String line;
            while ((line = br.readLine()) != null) {
                List<String> words = split(line);
                if (words.isEmpty())
                    continue;
                if (words.size() < 2)
                    throw error(lineNumber, words, "not enough words on line.");
                String type = words.get(0);
                if (type.equals("rule")) {
                    if (words.size() < 3)
                        throw error(lineNumber, words, "'rule' requires 2 arguments.");
                    output.addClassRename(new ClassRename(words.get(1), words.get(2)));
                } else if (type.equals("zap")) {
                    output.addClassDelete(new ClassDelete(words.get(1)));
                } else if (type.equals("keep")) {
                    output.addClassKeepTransitive(new ClassKeepTransitive(words.get(1)));
                } else {
                    throw error(lineNumber, words, "Unrecognized keyword " + type);
                }
                lineNumber++;
            }
        } finally {
            r.close();
        }
    }

    @Nonnull
    private static IllegalArgumentException error(@Nonnegative int lineNumber, @Nonnull List<String> words, @Nonnull String reason) {
        throw new IllegalArgumentException("Error on line " + lineNumber + ": " + words + ": " + reason);
    }
}
