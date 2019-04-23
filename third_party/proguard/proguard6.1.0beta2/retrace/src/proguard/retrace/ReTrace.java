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
package proguard.retrace;

import proguard.obfuscate.MappingReader;

import java.io.*;
import java.util.*;

/**
 * Tool for de-obfuscating stack traces of applications that were obfuscated
 * with ProGuard.
 *
 * @author Eric Lafortune
 */
public class ReTrace
{
    private static final String USAGE          = "Usage: java proguard.retrace.ReTrace [-regex <regex>] [-verbose] <mapping_file> [<stacktrace_file>]";
    private static final String REGEX_OPTION   = "-regex";
    private static final String VERBOSE_OPTION = "-verbose";

    public static final String STACK_TRACE_EXPRESSION = "(?:.*?\\bat\\s+%c\\.%m\\s*\\(%s(?::%l)?\\)\\s*(?:~\\[.*\\])?)|(?:(?:.*?[:\"]\\s+)?%c(?::.*)?)";


    // The settings.
    private final String  regularExpression;
    private final boolean verbose;
    private final File    mappingFile;


    /**
     * Creates a new ReTrace instance.
     * @param regularExpression the regular expression for parsing the lines in
     *                          the stack trace.
     * @param verbose           specifies whether the de-obfuscated stack trace
     *                          should be verbose.
     * @param mappingFile       the mapping file that was written out by
     *                          ProGuard.
     */
    public ReTrace(String  regularExpression,
                   boolean verbose,
                   File    mappingFile)
    {
        this.regularExpression = regularExpression;
        this.verbose           = verbose;
        this.mappingFile       = mappingFile;
    }


    /**
     * De-obfuscates a given stack trace.
     * @param stackTraceReader a reader for the obfuscated stack trace.
     * @param stackTraceWriter a writer for the de-obfuscated stack trace.
     */
    public void retrace(LineNumberReader stackTraceReader,
                        PrintWriter      stackTraceWriter) throws IOException
    {
        // Create a pattern for stack frames.
        FramePattern pattern = new FramePattern(regularExpression, verbose);

        // Create a remapper.
        FrameRemapper mapper = new FrameRemapper();

        // Read the mapping file.
        MappingReader mappingReader = new MappingReader(mappingFile);
        mappingReader.pump(mapper);

        // Read and process the lines of the stack trace.
        while (true)
        {
            // Read a line.
            String obfuscatedLine = stackTraceReader.readLine();
            if (obfuscatedLine == null)
            {
                break;
            }

            // Try to match it against the regular expression.
            FrameInfo obfuscatedFrame = pattern.parse(obfuscatedLine);
            if (obfuscatedFrame != null)
            {
                // Transform the obfuscated frame back to one or more
                // original frames.
                Iterator<FrameInfo> retracedFrames =
                    mapper.transform(obfuscatedFrame).iterator();

                String previousLine = null;

                while (retracedFrames.hasNext())
                {
                    // Retrieve the next retraced frame.
                    FrameInfo retracedFrame = retracedFrames.next();

                    // Format the retraced line.
                    String retracedLine =
                        pattern.format(obfuscatedLine, retracedFrame);

                    // Clear the common first part of ambiguous alternative
                    // retraced lines, to present a cleaner list of
                    // alternatives.
                    String trimmedLine =
                        previousLine != null &&
                        obfuscatedFrame.getLineNumber() == 0 ?
                            trim(retracedLine, previousLine) :
                            retracedLine;

                    // Print out the retraced line.
                    if (trimmedLine != null)
                    {
                        stackTraceWriter.println(trimmedLine);
                    }

                    previousLine = retracedLine;
                }
            }
            else
            {
                // Print out the original line.
                stackTraceWriter.println(obfuscatedLine);
            }
        }

        stackTraceWriter.flush();
    }


    /**
     * Returns the first given string, with any leading characters that it has
     * in common with the second string replaced by spaces.
     */
    private String trim(String string1, String string2)
    {
        StringBuilder line = new StringBuilder(string1);

        // Find the common part.
        int trimEnd = firstNonCommonIndex(string1, string2);
        if (trimEnd == string1.length())
        {
            return null;
        }

        // Don't clear the last identifier characters.
        trimEnd = lastNonIdentifierIndex(string1, trimEnd) + 1;

        // Clear the common characters.
        for (int index = 0; index < trimEnd; index++)
        {
            if (!Character.isWhitespace(string1.charAt(index)))
            {
                line.setCharAt(index, ' ');
            }
        }

        return line.toString();
    }


    /**
     * Returns the index of the first character that is not the same in both
     * given strings.
     */
    private int firstNonCommonIndex(String string1, String string2)
    {
        int index = 0;
        while (index < string1.length() &&
               index < string2.length() &&
               string1.charAt(index) == string2.charAt(index))
        {
            index++;
        }

        return index;
    }


    /**
     * Returns the index of the last character that is not an identifier
     * character in the given string, at or before the given index.
     */
    private int lastNonIdentifierIndex(String line, int index)
    {
        while (index >= 0 &&
               Character.isJavaIdentifierPart(line.charAt(index)))
        {
            index--;
        }

        return index;
    }


    /**
     * The main program for ReTrace.
     */
    public static void main(String[] args)
    {
        // Parse the arguments.
        if (args.length < 1)
        {
            System.err.println(USAGE);
            System.exit(-1);
        }

        String  regularExpresssion = STACK_TRACE_EXPRESSION;
        boolean verbose            = false;

        int argumentIndex = 0;
        while (argumentIndex < args.length)
        {
            String arg = args[argumentIndex];
            if (arg.equals(REGEX_OPTION))
            {
                regularExpresssion = args[++argumentIndex];
            }
            else if (arg.equals(VERBOSE_OPTION))
            {
                verbose = true;
            }
            else
            {
                break;
            }

            argumentIndex++;
        }

        if (argumentIndex >= args.length)
        {
            System.err.println(USAGE);
            System.exit(-1);
        }

        // Convert the arguments into File instances.
        File mappingFile    = new File(args[argumentIndex++]);
        File stackTraceFile = argumentIndex < args.length ?
            new File(args[argumentIndex]) :
            null;

        try
        {
            // Open the input stack trace. We're always using the UTF-8
            // character encoding, even for reading from the standard
            // input.
            LineNumberReader reader =
                new LineNumberReader(
                new BufferedReader(
                new InputStreamReader(stackTraceFile == null ? System.in :
                new FileInputStream(stackTraceFile), "UTF-8")));

            // Open the output stack trace, again using UTF-8 encoding.
            PrintWriter writer =
                new PrintWriter(new OutputStreamWriter(System.out, "UTF-8"));

            try
            {
                // Execute ReTrace with the collected settings.
                new ReTrace(regularExpresssion, verbose, mappingFile)
                    .retrace(reader, writer);
            }
            finally
            {
                // Close the input stack trace if it was a file.
                if (stackTraceFile != null)
                {
                    reader.close();
                }
            }
        }
        catch (IOException ex)
        {
            if (verbose)
            {
                // Print a verbose stack trace.
                ex.printStackTrace();
            }
            else
            {
                // Print just the stack trace message.
                System.err.println("Error: "+ex.getMessage());
            }

            System.exit(1);
        }

        System.exit(0);
    }
}
