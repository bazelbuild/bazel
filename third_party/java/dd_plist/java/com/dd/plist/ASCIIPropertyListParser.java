/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2014 Daniel Dreibrodt
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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.text.ParseException;
import java.text.StringCharacterIterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Parser for ASCII property lists. Supports Apple OS X/iOS and GnuStep/NeXTSTEP format.
 * This parser is based on the recursive descent paradigm, but the underlying grammar
 * is not explicitely defined.
 * <p/>
 * Resources on ASCII property list format:
 * <ul>
 * <li><a href="https://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/PropertyLists/OldStylePlists/OldStylePLists.html>
 * Property List Programming Guide - Old-Style ASCII Property Lists
 * </a></li>
 * <li><a href="http://www.gnustep.org/resources/documentation/Developer/Base/Reference/NSPropertyList.html">
 * GnuStep - NSPropertyListSerialization class documentation
 * </a></li>
 * </ul>
 *
 * @author Daniel Dreibrodt
 */
public class ASCIIPropertyListParser {

    /**
     * Parses an ASCII property list file.
     *
     * @param f The ASCII property list file.
     * @return The root object of the property list. This is usally a NSDictionary but can also be a NSArray.
     * @throws Exception When an error occurs during parsing.
     */
    public static NSObject parse(File f) throws IOException, ParseException {
        return parse(new FileInputStream(f));
    }

    /**
     * Parses an ASCII property list from an input stream.
     *
     * @param in The input stream that points to the property list's data.
     * @return The root object of the property list. This is usally a NSDictionary but can also be a NSArray.
     * @throws Exception When an error occurs during parsing.
     */
    public static NSObject parse(InputStream in) throws ParseException, IOException {
        byte[] buf = PropertyListParser.readAll(in);
        in.close();
        return parse(buf);
    }

    /**
     * Parses an ASCII property list from a byte array.
     *
     * @param bytes The ASCII property list data.
     * @return The root object of the property list. This is usally a NSDictionary but can also be a NSArray.
     * @throws Exception When an error occurs during parsing.
     */
    public static NSObject parse(byte[] bytes) throws ParseException {
        ASCIIPropertyListParser parser = new ASCIIPropertyListParser(bytes);
        return parser.parse();
    }

    public static final char WHITESPACE_SPACE = ' ';
    public static final char WHITESPACE_TAB = '\t';
    public static final char WHITESPACE_NEWLINE = '\n';
    public static final char WHITESPACE_CARRIAGE_RETURN = '\r';

    public static final char ARRAY_BEGIN_TOKEN = '(';
    public static final char ARRAY_END_TOKEN = ')';
    public static final char ARRAY_ITEM_DELIMITER_TOKEN = ',';

    public static final char DICTIONARY_BEGIN_TOKEN = '{';
    public static final char DICTIONARY_END_TOKEN = '}';
    public static final char DICTIONARY_ASSIGN_TOKEN = '=';
    public static final char DICTIONARY_ITEM_DELIMITER_TOKEN = ';';

    public static final char QUOTEDSTRING_BEGIN_TOKEN = '"';
    public static final char QUOTEDSTRING_END_TOKEN = '"';
    public static final char QUOTEDSTRING_ESCAPE_TOKEN = '\\';

    public static final char DATA_BEGIN_TOKEN = '<';
    public static final char DATA_END_TOKEN = '>';

    public static final char DATA_GSOBJECT_BEGIN_TOKEN = '*';
    public static final char DATA_GSDATE_BEGIN_TOKEN = 'D';
    public static final char DATA_GSBOOL_BEGIN_TOKEN = 'B';
    public static final char DATA_GSBOOL_TRUE_TOKEN = 'Y';
    public static final char DATA_GSBOOL_FALSE_TOKEN = 'N';
    public static final char DATA_GSINT_BEGIN_TOKEN = 'I';
    public static final char DATA_GSREAL_BEGIN_TOKEN = 'R';

    public static final char DATE_DATE_FIELD_DELIMITER = '-';
    public static final char DATE_TIME_FIELD_DELIMITER = ':';
    public static final char DATE_GS_DATE_TIME_DELIMITER = ' ';
    public static final char DATE_APPLE_DATE_TIME_DELIMITER = 'T';
    public static final char DATE_APPLE_END_TOKEN = 'Z';

    public static final char COMMENT_BEGIN_TOKEN = '/';
    public static final char MULTILINE_COMMENT_SECOND_TOKEN = '*';
    public static final char SINGLELINE_COMMENT_SECOND_TOKEN = '/';
    public static final char MULTILINE_COMMENT_END_TOKEN = '/';

    /**
     * Property list source data
     */
    private byte[] data;
    /**
     * Current parsing index
     */
    private int index;

    /**
     * Only allow subclasses to change instantiation.
     */
    protected ASCIIPropertyListParser() {

    }

    /**
     * Creates a new parser for the given property list content.
     *
     * @param propertyListContent The content of the property list that is to be parsed.
     */
    private ASCIIPropertyListParser(byte[] propertyListContent) {
        data = propertyListContent;
    }

    /**
     * Checks whether the given sequence of symbols can be accepted.
     *
     * @param sequence The sequence of tokens to look for.
     * @return Whether the given tokens occur at the current parsing position.
     */
    private boolean acceptSequence(char... sequence) {
        for (int i = 0; i < sequence.length; i++) {
            if (data[index + i] != sequence[i])
                return false;
        }
        return true;
    }

    /**
     * Checks whether the given symbols can be accepted, that is, if one
     * of the given symbols is found at the current parsing position.
     *
     * @param acceptableSymbols The symbols to check.
     * @return Whether one of the symbols can be accepted or not.
     */
    private boolean accept(char... acceptableSymbols) {
        boolean symbolPresent = false;
        for (char c : acceptableSymbols) {
            if (data[index] == c)
                symbolPresent = true;
        }
        return symbolPresent;
    }

    /**
     * Checks whether the given symbol can be accepted, that is, if
     * the given symbols is found at the current parsing position.
     *
     * @param acceptableSymbol The symbol to check.
     * @return Whether the symbol can be accepted or not.
     */
    private boolean accept(char acceptableSymbol) {
        return data[index] == acceptableSymbol;
    }

    /**
     * Expects the input to have one of the given symbols at the current parsing position.
     *
     * @param expectedSymbols The expected symbols.
     * @throws ParseException If none of the expected symbols could be found.
     */
    private void expect(char... expectedSymbols) throws ParseException {
        if (!accept(expectedSymbols)) {
            String excString = "Expected '" + expectedSymbols[0] + "'";
            for (int i = 1; i < expectedSymbols.length; i++) {
                excString += " or '" + expectedSymbols[i] + "'";
            }
            excString += " but found '" + (char) data[index] + "'";
            throw new ParseException(excString, index);
        }
    }

    /**
     * Expects the input to have the given symbol at the current parsing position.
     *
     * @param expectedSymbol The expected symbol.
     * @throws ParseException If the expected symbol could be found.
     */
    private void expect(char expectedSymbol) throws ParseException {
        if (!accept(expectedSymbol))
            throw new ParseException("Expected '" + expectedSymbol + "' but found '" + (char) data[index] + "'", index);
    }

    /**
     * Reads an expected symbol.
     *
     * @param symbol The symbol to read.
     * @throws ParseException If the expected symbol could not be read.
     */
    private void read(char symbol) throws ParseException {
        expect(symbol);
        index++;
    }

    /**
     * Skips the current symbol.
     */
    private void skip() {
        index++;
    }

    /**
     * Skips several symbols
     *
     * @param numSymbols The amount of symbols to skip.
     */
    private void skip(int numSymbols) {
        index += numSymbols;
    }

    /**
     * Skips all whitespaces and comments from the current parsing position onward.
     */
    private void skipWhitespacesAndComments() {
        boolean commentSkipped;
        do {
            commentSkipped = false;

            //Skip whitespaces
            while (accept(WHITESPACE_CARRIAGE_RETURN, WHITESPACE_NEWLINE, WHITESPACE_SPACE, WHITESPACE_TAB)) {
                skip();
            }

            //Skip single line comments "//..."
            if (acceptSequence(COMMENT_BEGIN_TOKEN, SINGLELINE_COMMENT_SECOND_TOKEN)) {
                skip(2);
                readInputUntil(WHITESPACE_CARRIAGE_RETURN, WHITESPACE_NEWLINE);
                commentSkipped = true;
            }
            //Skip multi line comments "/* ... */"
            else if (acceptSequence(COMMENT_BEGIN_TOKEN, MULTILINE_COMMENT_SECOND_TOKEN)) {
                skip(2);
                while (true) {
                    if (acceptSequence(MULTILINE_COMMENT_SECOND_TOKEN, MULTILINE_COMMENT_END_TOKEN)) {
                        skip(2);
                        break;
                    }
                    skip();
                }
                commentSkipped = true;
            }
        }
        while (commentSkipped); //if a comment was skipped more whitespace or another comment can follow, so skip again
    }

    private String toUtf8String(ByteArrayOutputStream stream) {
        try {
            return stream.toString("UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reads input until one of the given symbols is found.
     *
     * @param symbols The symbols that can occur after the string to read.
     * @return The input until one the given symbols.
     */
    private String readInputUntil(char... symbols) {
        ByteArrayOutputStream stringBytes = new ByteArrayOutputStream();
        while (!accept(symbols)) {
            stringBytes.write(data[index]);
            skip();
        }
        return toUtf8String(stringBytes);
    }

    /**
     * Reads input until the given symbol is found.
     *
     * @param symbol The symbol that can occur after the string to read.
     * @return The input until the given symbol.
     */
    private String readInputUntil(char symbol) {
        ByteArrayOutputStream stringBytes = new ByteArrayOutputStream();
        while (!accept(symbol)) {
            stringBytes.write(data[index]);
            skip();
        }
        return toUtf8String(stringBytes);
    }

    /**
     * Parses the property list from the beginning and returns the root object
     * of the property list.
     *
     * @return The root object of the property list. This can either be a NSDictionary or a NSArray.
     * @throws ParseException When an error occured during parsing
     */
    public NSObject parse() throws ParseException {
        index = 0;
        skipWhitespacesAndComments();
        expect(DICTIONARY_BEGIN_TOKEN, ARRAY_BEGIN_TOKEN, COMMENT_BEGIN_TOKEN);
        try {
            return parseObject();
        } catch (ArrayIndexOutOfBoundsException ex) {
            throw new ParseException("Reached end of input unexpectedly.", index);
        }
    }

    /**
     * Parses the NSObject found at the current position in the property list
     * data stream.
     *
     * @return The parsed NSObject.
     * @see ASCIIPropertyListParser#index
     */
    private NSObject parseObject() throws ParseException {
        switch (data[index]) {
            case ARRAY_BEGIN_TOKEN: {
                return parseArray();
            }
            case DICTIONARY_BEGIN_TOKEN: {
                return parseDictionary();
            }
            case DATA_BEGIN_TOKEN: {
                return parseData();
            }
            case QUOTEDSTRING_BEGIN_TOKEN: {
                String quotedString = parseQuotedString();
                //apple dates are quoted strings of length 20 and after the 4 year digits a dash is found
                if (quotedString.length() == 20 && quotedString.charAt(4) == DATE_DATE_FIELD_DELIMITER) {
                    try {
                        return new NSDate(quotedString);
                    } catch (Exception ex) {
                        //not a date? --> return string
                        return new NSString(quotedString);
                    }
                } else {
                    return new NSString(quotedString);
                }
            }
            default: {
                //0-9
                if (data[index] > 0x2F && data[index] < 0x3A) {
                    //could be a date or just a string
                    return parseDateString();
                } else {
                    //non-numerical -> string or boolean
                    String parsedString = parseString();
                    return new NSString(parsedString);
                }
            }
        }
    }

    /**
     * Parses an array from the current parsing position.
     * The prerequisite for calling this method is, that an array begin token has been read.
     *
     * @return The array found at the parsing position.
     */
    private NSArray parseArray() throws ParseException {
        //Skip begin token
        skip();
        skipWhitespacesAndComments();
        List<NSObject> objects = new LinkedList<NSObject>();
        while (!accept(ARRAY_END_TOKEN)) {
            objects.add(parseObject());
            skipWhitespacesAndComments();
            if (accept(ARRAY_ITEM_DELIMITER_TOKEN)) {
                skip();
            } else {
                break; //must have reached end of array
            }
            skipWhitespacesAndComments();
        }
        //parse end token
        read(ARRAY_END_TOKEN);
        return new NSArray(objects.toArray(new NSObject[objects.size()]));
    }

    /**
     * Parses a dictionary from the current parsing position.
     * The prerequisite for calling this method is, that a dictionary begin token has been read.
     *
     * @return The dictionary found at the parsing position.
     */
    private NSDictionary parseDictionary() throws ParseException {
        //Skip begin token
        skip();
        skipWhitespacesAndComments();
        NSDictionary dict = new NSDictionary();
        while (!accept(DICTIONARY_END_TOKEN)) {
            //Parse key
            String keyString;
            if (accept(QUOTEDSTRING_BEGIN_TOKEN)) {
                keyString = parseQuotedString();
            } else {
                keyString = parseString();
            }
            skipWhitespacesAndComments();

            //Parse assign token
            read(DICTIONARY_ASSIGN_TOKEN);
            skipWhitespacesAndComments();

            NSObject object = parseObject();
            dict.put(keyString, object);
            skipWhitespacesAndComments();
            read(DICTIONARY_ITEM_DELIMITER_TOKEN);
            skipWhitespacesAndComments();
        }
        //skip end token
        skip();
        return dict;
    }

    /**
     * Parses a data object from the current parsing position.
     * This can either be a NSData object or a GnuStep NSNumber or NSDate.
     * The prerequisite for calling this method is, that a data begin token has been read.
     *
     * @return The data object found at the parsing position.
     */
    private NSObject parseData() throws ParseException {
        NSObject obj = null;
        //Skip begin token
        skip();
        if (accept(DATA_GSOBJECT_BEGIN_TOKEN)) {
            skip();
            expect(DATA_GSBOOL_BEGIN_TOKEN, DATA_GSDATE_BEGIN_TOKEN, DATA_GSINT_BEGIN_TOKEN, DATA_GSREAL_BEGIN_TOKEN);
            if (accept(DATA_GSBOOL_BEGIN_TOKEN)) {
                //Boolean
                skip();
                expect(DATA_GSBOOL_TRUE_TOKEN, DATA_GSBOOL_FALSE_TOKEN);
                if (accept(DATA_GSBOOL_TRUE_TOKEN)) {
                    obj = new NSNumber(true);
                } else {
                    obj = new NSNumber(false);
                }
                //Skip the parsed boolean token
                skip();
            } else if (accept(DATA_GSDATE_BEGIN_TOKEN)) {
                //Date
                skip();
                String dateString = readInputUntil(DATA_END_TOKEN);
                obj = new NSDate(dateString);
            } else if (accept(DATA_GSINT_BEGIN_TOKEN, DATA_GSREAL_BEGIN_TOKEN)) {
                //Number
                skip();
                String numberString = readInputUntil(DATA_END_TOKEN);
                obj = new NSNumber(numberString);
            }
            //parse data end token
            read(DATA_END_TOKEN);
        } else {
            String dataString = readInputUntil(DATA_END_TOKEN);
            dataString = dataString.replaceAll("\\s+", "");

            int numBytes = dataString.length() / 2;
            byte[] bytes = new byte[numBytes];
            for (int i = 0; i < bytes.length; i++) {
                String byteString = dataString.substring(i * 2, i * 2 + 2);
                int byteValue = Integer.parseInt(byteString, 16);
                bytes[i] = (byte) byteValue;
            }
            obj = new NSData(bytes);

            //skip end token
            skip();
        }

        return obj;
    }

    /**
     * Attempts to parse a plain string as a date if possible.
     *
     * @return A NSDate if the string represents such an object. Otherwise a NSString is returned.
     */
    private NSObject parseDateString() {
        String numericalString = parseString();
        if (numericalString.length() > 4 && numericalString.charAt(4) == DATE_DATE_FIELD_DELIMITER) {
            try {
                return new NSDate(numericalString);
            } catch(Exception ex) {
                //An exception occurs if the string is not a date but just a string
            }
        }
        return new NSString(numericalString);
    }

    /**
     * Parses a plain string from the current parsing position.
     * The string is made up of all characters to the next whitespace, delimiter token or assignment token.
     *
     * @return The string found at the current parsing position.
     */
    private String parseString() {
        return readInputUntil(WHITESPACE_SPACE, WHITESPACE_TAB, WHITESPACE_NEWLINE, WHITESPACE_CARRIAGE_RETURN,
                ARRAY_ITEM_DELIMITER_TOKEN, DICTIONARY_ITEM_DELIMITER_TOKEN, DICTIONARY_ASSIGN_TOKEN, ARRAY_END_TOKEN);
    }

    /**
     * Parses a quoted string from the current parsing position.
     * The prerequisite for calling this method is, that a quoted string begin token has been read.
     *
     * @return The quoted string found at the parsing method with all special characters unescaped.
     * @throws ParseException If an error occured during parsing.
     */
    private String parseQuotedString() throws ParseException {
        //Skip begin token
        skip();
        ByteArrayOutputStream quotedString = new ByteArrayOutputStream();
        boolean unescapedBackslash = true;
        //Read from opening quotation marks to closing quotation marks and skip escaped quotation marks
        while (data[index] != QUOTEDSTRING_END_TOKEN || (data[index - 1] == QUOTEDSTRING_ESCAPE_TOKEN && unescapedBackslash)) {
            quotedString.write(data[index]);
            if (accept(QUOTEDSTRING_ESCAPE_TOKEN)) {
                unescapedBackslash = !(data[index - 1] == QUOTEDSTRING_ESCAPE_TOKEN && unescapedBackslash);
            }
            skip();
        }
        String unescapedString;
        try {
            unescapedString = parseQuotedString(toUtf8String(quotedString));
        } catch (Exception ex) {
            throw new ParseException("The quoted string could not be parsed.", index);
        }
        //skip end token
        skip();
        return unescapedString;
    }

    /**
     * Used to encode the parsed strings
     */
    private static CharsetEncoder asciiEncoder;

    /**
     * Parses a string according to the format specified for ASCII property lists.
     * Such strings can contain escape sequences which are unescaped in this method.
     *
     * @param s The escaped string according to the ASCII property list format, without leading and trailing quotation marks.
     * @return The unescaped string in UTF-8 or ASCII format, depending on the contained characters.
     * @throws Exception If the string could not be properly parsed.
     */
    public static synchronized String parseQuotedString(String s) throws UnsupportedEncodingException, CharacterCodingException {
        StringBuilder parsed = new StringBuilder();
        StringCharacterIterator iterator = new StringCharacterIterator(s);
        char c = iterator.current();

        while (iterator.getIndex() < iterator.getEndIndex()) {
            switch (c) {
                case '\\': { //An escaped sequence is following
                    parsed.append(parseEscapedSequence(iterator));
                    break;
                }
                default: {
                    parsed.append(c);
                    break;
                }
            }
            c = iterator.next();
        }
        return parsed.toString();
    }

    /**
     * Unescapes an escaped character sequence, e.g. \\u00FC.
     *
     * @param iterator The string character iterator pointing to the first character after the backslash
     * @return The unescaped character
     */
    private static char parseEscapedSequence(StringCharacterIterator iterator) {
        char c = iterator.next();
        if (c == 'b') {
            return '\b';
        } else if (c == 'n') {
            return '\n';
        } else if (c == 'r') {
            return '\r';
        } else if (c == 't') {
            return '\t';
        } else if (c == 'U' || c == 'u') {
            //4 digit hex Unicode value
            String byte1 = "";
            byte1 += iterator.next();
            byte1 += iterator.next();
            String byte2 = "";
            byte2 += iterator.next();
            byte2 += iterator.next();
            return (char) ((Integer.parseInt(byte1, 16) << 8) + Integer.parseInt(byte2, 16));
        } else if ((c >= '0') && (c <= '7')) {
            //3 digit octal ASCII value
            String num = "";
            num += c;
            num += iterator.next();
            num += iterator.next();
            return (char) Integer.parseInt(num, 8);
        } else {
            // Possibly something that needn't be escaped, but we should accept it
            // it anyway for consistency with Apple tools.
            return c;
        }
    }

}
