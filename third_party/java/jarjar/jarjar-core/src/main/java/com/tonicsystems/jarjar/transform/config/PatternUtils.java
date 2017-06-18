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

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nonnull;

public class PatternUtils {

    private PatternUtils() {
    }

    private static final Pattern dstar = Pattern.compile("\\*\\*");
    private static final Pattern star = Pattern.compile("\\*");
    private static final Pattern estar = Pattern.compile("\\+\\??\\)\\Z");

    @Nonnull
    private static String replaceAllLiteral(@Nonnull String value, @Nonnull Pattern pattern, @Nonnull String replace) {
        return pattern.matcher(value).replaceAll(Matcher.quoteReplacement(replace));
    }

    @Nonnull
    public static Pattern newPattern(@Nonnull String pattern) {
        if (pattern.equals("**"))
            throw new IllegalArgumentException("'**' is not a valid pattern");
        if (!isPossibleQualifiedName(pattern, "/*"))
            throw new IllegalArgumentException("Not a valid package pattern: " + pattern);
        if (pattern.indexOf("***") >= 0)
            throw new IllegalArgumentException("The sequence '***' is invalid in a package pattern");

        String regex = pattern;
        regex = replaceAllLiteral(regex, dstar, "(.+?)");   // One wildcard test requires the argument to be allowably empty.
        regex = replaceAllLiteral(regex, star, "([^/]+)");
        regex = replaceAllLiteral(regex, estar, "*\\??)");  // Although we replaced with + above, we mean *
        return Pattern.compile("\\A" + regex + "\\Z");
        // this.count = this.pattern.matcher("foo").groupCount();
    }

    private static enum State {

        NORMAL, ESCAPE;
    }

    @Nonnull
    public static List<Object> newReplace(@Nonnull Pattern pattern, @Nonnull String result) {
        List<Object> parts = new ArrayList<Object>(16);
        // TODO: check for illegal characters
        int max = 0;
        State state = State.NORMAL;
        for (int i = 0, mark = 0, len = result.length(); i < len + 1; i++) {
            char ch = (i == len) ? '@' : result.charAt(i);
            switch (state) {
                case NORMAL:
                    if (ch == '@') {
                        parts.add(result.substring(mark, i).replace('.', '/'));
                        mark = i + 1;
                        state = State.ESCAPE;
                    }
                    break;
                case ESCAPE:
                    switch (ch) {
                        case '0':
                        case '1':
                        case '2':
                        case '3':
                        case '4':
                        case '5':
                        case '6':
                        case '7':
                        case '8':
                        case '9':
                            break;
                        default:
                            if (i == mark)
                                throw new IllegalArgumentException("Backslash not followed by a digit");
                            int n = Integer.parseInt(result.substring(mark, i));
                            if (n > max)
                                max = n;
                            parts.add(Integer.valueOf(n));
                            mark = i--;
                            state = State.NORMAL;
                            break;
                    }
                    break;
            }
        }

        int count = pattern.matcher("foo").groupCount();
        if (count < max)
            throw new IllegalArgumentException("Result includes impossible placeholder \"@" + max + "\": " + result);
        // System.err.println(this);
        return parts;
    }

    public static String replace(@Nonnull AbstractPattern pattern, @Nonnull List<Object> replace, String value) {
        Matcher matcher = pattern.getMatcher(value);
        if (matcher == null)
            return null;
        StringBuilder sb = new StringBuilder();
        for (Object part : replace) {
            if (part instanceof String)
                sb.append((String) part);
            else
                sb.append(matcher.group((Integer) part));
        }
        return sb.toString();
    }

    public static final String PACKAGE_INFO = "package-info";

    /* pp */ static boolean isPossibleQualifiedName(@Nonnull String value, @Nonnull String extraAllowedCharacters) {
        // package-info violates the spec for Java Identifiers.
        // Nevertheless, expressions that end with this string are still legal.
        // See 7.4.1.1 of the Java language spec for discussion.
        if (value.endsWith(PACKAGE_INFO)) {
            value = value.substring(0, value.length() - PACKAGE_INFO.length());
        }
        for (int i = 0, len = value.length(); i < len; i++) {
            char c = value.charAt(i);
            if (Character.isJavaIdentifierPart(c))
                continue;
            if (extraAllowedCharacters.indexOf(c) >= 0)
                continue;
            return false;
        }
        return true;
    }

    /**
     * Copies the given {@link Iterable} into a new {@link List}.
     *
     * @param <T> The free parameter for the element type.
     * @param in The Iterable to copy.
     * @return A new, mutable {@link ArrayList}.
     */
    @Nonnull
    public static <T extends AbstractPattern> List<T> toList(@Nonnull Iterable<? extends T> in) {
        List<T> out = new ArrayList<T>();
        for (T i : in)
            out.add(i);
        return out;
    }

    // Adapted from http://stackoverflow.com/questions/1247772/is-there-an-equivalent-of-java-util-regex-for-glob-type-patterns
    @Nonnull
    public static String convertGlobToRegEx(@Nonnull String line) {
        line = line.trim();
        int strLen = line.length();
        StringBuilder sb = new StringBuilder(strLen);
        // Remove beginning and ending * globs because they're useless
        if (line.startsWith("*")) {
            line = line.substring(1);
            strLen--;
        }
        if (line.endsWith("*")) {
            line = line.substring(0, strLen - 1);
            strLen--;
        }
        boolean escaping = false;
        int inCurlies = 0;
        CHAR:
        for (char currentChar : line.toCharArray()) {
            switch (currentChar) {
                case '*':
                    if (escaping)
                        sb.append("\\*");
                    else
                        sb.append(".*");
                    break;
                case '?':
                    if (escaping)
                        sb.append("\\?");
                    else
                        sb.append('.');
                    break;
                case '.':
                case '(':
                case ')':
                case '+':
                case '|':
                case '^':
                case '$':
                case '@':
                case '%':
                    sb.append('\\');
                    sb.append(currentChar);
                    break;
                case '\\':
                    if (escaping)
                        sb.append("\\\\");
                    else {
                        escaping = true;
                        continue CHAR;
                    }
                    break;
                case '{':
                    if (escaping)
                        sb.append("\\{");
                    else {
                        sb.append('(');
                        inCurlies++;
                    }
                    break;
                case '}':
                    if (escaping)
                        sb.append("\\}");
                    else if (inCurlies > 0) {
                        sb.append(')');
                        inCurlies--;
                    } else
                        sb.append("}");
                    break;
                case ',':
                    if (escaping)
                        sb.append("\\,");
                    else if (inCurlies > 0)
                        sb.append('|');
                    else
                        sb.append(",");
                    break;
                default:
                    sb.append(currentChar);
                    break;
            }
            escaping = false;
        }
        return sb.toString();
    }
}
