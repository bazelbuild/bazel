/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard;

import java.io.*;
import java.net.URL;


/**
 * An abstract reader of words, with the possibility to include other readers.
 * Words are separated by spaces or broken off at delimiters. Words containing
 * spaces or delimiters can be quoted with single or double quotes.
 * Comments (everything starting with '#' on a single line) are ignored.
 *
 * @author Eric Lafortune
 * @noinspection TailRecursion
 */
public abstract class WordReader
{
    private static final char COMMENT_CHARACTER = '#';


    private File       baseDir;
    private URL        baseURL;
    private WordReader includeWordReader;
    private String     currentLine;
    private int        currentLineLength;
    private int        currentIndex;
    private String     currentWord;
    private String     currentComments;


    /**
     * Creates a new WordReader with the given base directory.
     */
    protected WordReader(File baseDir)
    {
        this.baseDir = baseDir;
    }


    /**
     * Creates a new WordReader with the given base URL.
     */
    protected WordReader(URL baseURL)
    {
        this.baseURL = baseURL;
    }


    /**
     * Sets the base directory of this reader.
     */
    public void setBaseDir(File baseDir)
    {
        if (includeWordReader != null)
        {
            includeWordReader.setBaseDir(baseDir);
        }
        else
        {
            this.baseDir = baseDir;
        }
    }


    /**
     * Returns the base directory of this reader, if any.
     */
    public File getBaseDir()
    {
        return includeWordReader != null ?
            includeWordReader.getBaseDir() :
            baseDir;
    }


    /**
     * Returns the base URL of this reader, if any.
     */
    public URL getBaseURL()
    {
        return includeWordReader != null ?
            includeWordReader.getBaseURL() :
            baseURL;
    }


    /**
     * Specifies to start reading words from the given WordReader. When it is
     * exhausted, this WordReader will continue to provide its own words.
     *
     * @param newIncludeWordReader the WordReader that will start reading words.
     */
    public void includeWordReader(WordReader newIncludeWordReader)
    {
        if (includeWordReader == null)
        {
            includeWordReader = newIncludeWordReader;
        }
        else
        {
            includeWordReader.includeWordReader(newIncludeWordReader);
        }
    }


    /**
     * Reads a word from this WordReader, or from one of its active included
     * WordReader objects.
     *
     * @param isFileName         return a complete line (or argument), if the word
     *                           isn't an option (it doesn't start with '-').
     * @param expectSingleFile   if true, the remaining line is expected to be a
     *                           single file name (excluding path separator),
     *                           otherwise multiple files might be specified
     *                           using the path separator.
     * @return the read word.
     */
    public String nextWord(boolean isFileName,
                           boolean expectSingleFile) throws IOException
    {
        currentWord = null;

        // See if we have an included reader to produce a word.
        if (includeWordReader != null)
        {
            // Does the included word reader still produce a word?
            currentWord = includeWordReader.nextWord(isFileName, expectSingleFile);
            if (currentWord != null)
            {
                // Return it if so.
                return currentWord;
            }

            // Otherwise close and ditch the word reader.
            includeWordReader.close();
            includeWordReader = null;
        }

        // Get a word from this reader.

        // Skip any whitespace and comments left on the current line.
        if (currentLine != null)
        {
            // Skip any leading whitespace.
            while (currentIndex < currentLineLength &&
                   Character.isWhitespace(currentLine.charAt(currentIndex)))
            {
                currentIndex++;
            }

            // Skip any comments.
            if (currentIndex < currentLineLength &&
                isComment(currentLine.charAt(currentIndex)))
            {
                currentIndex = currentLineLength;
            }
        }

        // Make sure we have a non-blank line.
        while (currentLine == null || currentIndex == currentLineLength)
        {
            currentLine = nextLine();
            if (currentLine == null)
            {
                return null;
            }

            currentLineLength = currentLine.length();

            // Skip any leading whitespace.
            currentIndex = 0;
            while (currentIndex < currentLineLength &&
                   Character.isWhitespace(currentLine.charAt(currentIndex)))
            {
                currentIndex++;
            }

            // Remember any leading comments.
            if (currentIndex < currentLineLength &&
                isComment(currentLine.charAt(currentIndex)))
            {
                // Remember the comments.
                String comment = currentLine.substring(currentIndex + 1);
                currentComments = currentComments == null ?
                    comment :
                    currentComments + '\n' + comment;

                // Skip the comments.
                currentIndex = currentLineLength;
            }
        }

        // Find the word starting at the current index.
        int startIndex = currentIndex;
        int endIndex;

        char startChar = currentLine.charAt(startIndex);

        if (isQuote(startChar))
        {
            // The next word is starting with a quote character.
            // Skip the opening quote.
            startIndex++;

            // The next word is a quoted character string.
            // Find the closing quote.
            do
            {
                currentIndex++;

                if (currentIndex == currentLineLength)
                {
                    currentWord = currentLine.substring(startIndex-1, currentIndex);
                    throw new IOException("Missing closing quote for "+locationDescription());
                }
            }
            while (currentLine.charAt(currentIndex) != startChar);

            endIndex = currentIndex++;
        }
        else if (isFileName &&
                 !isOption(startChar))
        {
            // The next word is a (possibly optional) file name.
            // Find the end of the line, the first path separator, the first
            // option, or the first comment.
            while (currentIndex < currentLineLength)
            {
                char currentCharacter = currentLine.charAt(currentIndex);
                if (isFileDelimiter(currentCharacter, !expectSingleFile) ||
                    ((isOption(currentCharacter) ||
                      isComment(currentCharacter)) &&
                     Character.isWhitespace(currentLine.charAt(currentIndex-1)))) {
                    break;
                }

                currentIndex++;
            }

            endIndex = currentIndex;

            // Trim any trailing whitespace.
            while (endIndex > startIndex &&
                   Character.isWhitespace(currentLine.charAt(endIndex-1)))
            {
                endIndex--;
            }
        }
        else if (isDelimiter(startChar))
        {
            // The next word is a single delimiting character.
            endIndex = ++currentIndex;
        }
        else
        {
            // The next word is a simple character string.
            // Find the end of the line, the first delimiter, or the first
            // white space.
            while (currentIndex < currentLineLength)
            {
                char currentCharacter = currentLine.charAt(currentIndex);
                if (isNonStartDelimiter(currentCharacter)    ||
                    Character.isWhitespace(currentCharacter) ||
                    isComment(currentCharacter)) {
                    break;
                }

                currentIndex++;
            }

            endIndex = currentIndex;
        }

        // Remember and return the parsed word.
        currentWord = currentLine.substring(startIndex, endIndex);

        return currentWord;
    }


    /**
     * Returns the comments collected before returning the last word.
     * Starts collecting new comments.
     *
     * @return the collected comments, or <code>null</code> if there weren't any.
     */
    public String lastComments() throws IOException
    {
        if (includeWordReader == null)
        {
            String comments = currentComments;
            currentComments = null;
            return comments;
        }
        else
        {
            return includeWordReader.lastComments();
        }
    }


    /**
     * Constructs a readable description of the current position in this
     * WordReader and its included WordReader objects.
     *
     * @return the description.
     */
    public String locationDescription()
    {
        return
            (includeWordReader == null ?
                (currentWord == null ?
                    "end of " :
                    "'" + currentWord + "' in " ) :
                (includeWordReader.locationDescription() + ",\n" +
                 "  included from ")) +
            lineLocationDescription();
    }


    /**
     * Reads a line from this WordReader, or from one of its active included
     * WordReader objects.
     *
     * @return the read line.
     */
    protected abstract String nextLine() throws IOException;


    /**
     * Returns a readable description of the current WordReader position.
     *
     * @return the description.
     */
    protected abstract String lineLocationDescription();


    /**
     * Closes the FileWordReader.
     */
    public void close() throws IOException
    {
        // Close and ditch the included word reader, if any.
        if (includeWordReader != null)
        {
            includeWordReader.close();
            includeWordReader = null;
        }
    }


    // Small utility methods.

    private boolean isOption(char character)
    {
        return character == '-';
    }


    private boolean isComment(char character)
    {
        return character == COMMENT_CHARACTER;
    }


    private boolean isDelimiter(char character)
    {
        return isStartDelimiter(character) || isNonStartDelimiter(character);
    }


    private boolean isStartDelimiter(char character)
    {
        return character == '@';
    }


    private boolean isNonStartDelimiter(char character)
    {
        return character == '{' ||
               character == '}' ||
               character == '(' ||
               character == ')' ||
               character == ',' ||
               character == ';' ||
               character == File.pathSeparatorChar;
    }


    private boolean isFileDelimiter(char    character,
                                    boolean includePathSeparator)
    {
        return character == '(' ||
               character == ')' ||
               character == ',' ||
               character == ';' ||
               (includePathSeparator &&
                character == File.pathSeparatorChar);
    }


    private boolean isQuote(char character)
    {
        return character == '\'' ||
               character == '"';
    }
}
