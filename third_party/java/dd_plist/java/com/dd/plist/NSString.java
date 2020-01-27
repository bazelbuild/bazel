/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2011 Daniel Dreibrodt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.dd.plist;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;

/**
 * A NSString contains a string.
 *
 * @author Daniel Dreibrodt
 */
public class NSString extends NSObject implements Comparable<Object> {

    private String content;

    /**
     * Creates an NSString from its binary representation.
     *
     * @param bytes    The binary representation.
     * @param encoding The encoding of the binary representation, the name of a supported charset.
     * @throws UnsupportedEncodingException When the given encoding is not supported by the JRE.
     * @see java.lang.String#String(byte[], String)
     */
    public NSString(byte[] bytes, String encoding) throws UnsupportedEncodingException {
        this(bytes, 0, bytes.length, encoding);
    }

    /**
     * Creates an NSString from its binary representation.
     *
     * @param bytes The binary representation.
     * @param startIndex int with the index where to start (offset)
     * @param endIndex int with the index where to stop reading (offset + string length)
     * @param encoding The encoding of the binary representation, the name of a supported charset.
     * @throws UnsupportedEncodingException When the given encoding is not supported by the JRE.
     * @see java.lang.String#String(byte[], String)
     */
    public NSString(byte[] bytes, final int startIndex, final int endIndex, String encoding) throws UnsupportedEncodingException {
        content = new String(bytes, startIndex, endIndex - startIndex, encoding);
    }

    /**
     * Creates a NSString from a string.
     *
     * @param string The string that will be contained in the NSString.
     */
    public NSString(String string) {
        content = string;
    }

    /**
     * Gets this strings content.
     *
     * @return This NSString as Java String object.
     */
    public String getContent() {
        return content;
    }

    /**
     * Sets the contents of this string.
     *
     * @param c The new content of this string object.
     */
    public void setContent(String c) {
        content = c;
    }

    /**
     * Appends a string to this string.
     *
     * @param s The string to append.
     */
    public void append(NSString s) {
        append(s.getContent());
    }

    /**
     * Appends a string to this string.
     *
     * @param s The string to append.
     */
    public void append(String s) {
        content += s;
    }

    /**
     * Prepends a string to this string.
     *
     * @param s The string to prepend.
     */
    public void prepend(String s) {
        content = s + content;
    }

    /**
     * Prepends a string to this string.
     *
     * @param s The string to prepend.
     */
    public void prepend(NSString s) {
        prepend(s.getContent());
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof NSString)) return false;
        return content.equals(((NSString) obj).content);
    }

    @Override
    public int hashCode() {
        return content.hashCode();
    }

    /**
     * The textual representation of this NSString.
     *
     * @return The NSString's contents.
     */
    @Override
    public String toString() {
        return content;
    }

    private static CharsetEncoder asciiEncoder, utf16beEncoder, utf8Encoder;

    @Override
    void toXML(StringBuilder xml, int level) {
        indent(xml, level);
        xml.append("<string>");

        //Make sure that the string is encoded in UTF-8 for the XML output
        synchronized (NSString.class) {
            if (utf8Encoder == null)
                utf8Encoder = Charset.forName("UTF-8").newEncoder();
            else
                utf8Encoder.reset();

            try {
                ByteBuffer byteBuf = utf8Encoder.encode(CharBuffer.wrap(content));
                byte[] bytes = new byte[byteBuf.remaining()];
                byteBuf.get(bytes);
                content = new String(bytes, "UTF-8");
            } catch (Exception ex) {
                throw new RuntimeException("Could not encode the NSString into UTF-8: " + String.valueOf(ex.getMessage()));
            }
        }

        //According to http://www.w3.org/TR/REC-xml/#syntax node values must not
        //contain the characters < or &. Also the > character should be escaped.
        if (content.contains("&") || content.contains("<") || content.contains(">")) {
            xml.append("<![CDATA[");
            xml.append(content.replaceAll("]]>", "]]]]><![CDATA[>"));
            xml.append("]]>");
        } else {
            xml.append(content);
        }
        xml.append("</string>");
    }


    @Override
    public void toBinary(BinaryPropertyListWriter out) throws IOException {
        CharBuffer charBuf = CharBuffer.wrap(content);
        int kind;
        ByteBuffer byteBuf;
        synchronized (NSString.class) {
            if (asciiEncoder == null)
                asciiEncoder = Charset.forName("ASCII").newEncoder();
            else
                asciiEncoder.reset();

            if (asciiEncoder.canEncode(charBuf)) {
                kind = 0x5; // standard ASCII
                byteBuf = asciiEncoder.encode(charBuf);
            } else {
                if (utf16beEncoder == null)
                    utf16beEncoder = Charset.forName("UTF-16BE").newEncoder();
                else
                    utf16beEncoder.reset();

                kind = 0x6; // UTF-16-BE
                byteBuf = utf16beEncoder.encode(charBuf);
            }
        }
        byte[] bytes = new byte[byteBuf.remaining()];
        byteBuf.get(bytes);
        out.writeIntHeader(kind, content.length());
        out.write(bytes);
    }

    @Override
    protected void toASCII(StringBuilder ascii, int level) {
        indent(ascii, level);
        ascii.append("\"");
        //According to https://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/PropertyLists/OldStylePlists/OldStylePLists.html
        //non-ASCII characters are not escaped but simply written into the
        //file, thus actually violating the ASCII plain text format.
        //We will escape the string anyway because current Xcode project files (ASCII property lists) also escape their strings.
        ascii.append(escapeStringForASCII(content));
        ascii.append("\"");
    }

    @Override
    protected void toASCIIGnuStep(StringBuilder ascii, int level) {
        indent(ascii, level);
        ascii.append("\"");
        ascii.append(escapeStringForASCII(content));
        ascii.append("\"");
    }

    /**
     * Escapes a string for use in ASCII property lists.
     *
     * @param s The unescaped string.
     * @return The escaped string.
     */
    static String escapeStringForASCII(String s) {
        String out = "";
        char[] cArray = s.toCharArray();
        for (int i = 0; i < cArray.length; i++) {
            char c = cArray[i];
            if (c > 127) {
                //non-ASCII Unicode
                out += "\\U";
                String hex = Integer.toHexString(c);
                while (hex.length() < 4)
                    hex = "0" + hex;
                out += hex;
            } else if (c == '\\') {
                out += "\\\\";
            } else if (c == '\"') {
                out += "\\\"";
            } else if (c == '\b') {
                out += "\\b";
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\r') {
                out += "\\r";
            } else if (c == '\t') {
                out += "\\t";
            } else {
                out += c;
            }
        }
        return out;
    }

    public int compareTo(Object o) {
        if (o instanceof NSString) {
            return getContent().compareTo(((NSString) o).getContent());
        } else if (o instanceof String) {
            return getContent().compareTo(((String) o));
        } else {
            return -1;
        }
    }
}
