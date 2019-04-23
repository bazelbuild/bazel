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

import proguard.classfile.util.ClassUtil;

import java.util.regex.*;

/**
 * This class can parse and format lines that represent stack frames
 * matching a given regular expression.
 *
 * @author Eric Lafortune
 */
public class FramePattern
{
    // The pattern matcher has problems with \\b against some unicode
    // characters, so we're no longer using \\b for classes and class members.
    private static final String REGEX_CLASS       = "(?:[^\\s\":./()]+\\.)*[^\\s\":./()]+";
    private static final String REGEX_CLASS_SLASH = "(?:[^\\s\":./()]+/)*[^\\s\":./()]+";
    private static final String REGEX_SOURCE_FILE = "[^:()]*";
    private static final String REGEX_LINE_NUMBER = "-?\\b\\d+\\b";
    private static final String REGEX_TYPE        = REGEX_CLASS + "(?:\\[\\])*";
    private static final String REGEX_MEMBER      = "<?[^\\s\":./()]+>?";
    private static final String REGEX_ARGUMENTS   = "(?:" + REGEX_TYPE + "(?:\\s*,\\s*" + REGEX_TYPE + ")*)?";

    private final char[]   expressionTypes     = new char[32];
    private final int      expressionTypeCount;
    private final Pattern  pattern;
    private final boolean  verbose;


    /**
     * Creates a new FramePattern.
     */
    public FramePattern(String regularExpression, boolean verbose)
    {
        // Construct the regular expression.
        StringBuffer expressionBuffer = new StringBuffer(regularExpression.length() + 32);

        int expressionTypeCount = 0;
        int index = 0;
        while (true)
        {
            int nextIndex = regularExpression.indexOf('%', index);
            if (nextIndex < 0                             ||
                nextIndex == regularExpression.length()-1 ||
                expressionTypeCount == expressionTypes.length)
            {
                break;
            }

            // Copy a literal piece of the input line.
            expressionBuffer.append(regularExpression.substring(index, nextIndex));
            expressionBuffer.append('(');

            char expressionType = regularExpression.charAt(nextIndex + 1);
            switch(expressionType)
            {
                case 'c':
                    expressionBuffer.append(REGEX_CLASS);
                    break;

                case 'C':
                    expressionBuffer.append(REGEX_CLASS_SLASH);
                    break;

                case 's':
                    expressionBuffer.append(REGEX_SOURCE_FILE);
                    break;

                case 'l':
                    expressionBuffer.append(REGEX_LINE_NUMBER);
                    break;

                case 't':
                    expressionBuffer.append(REGEX_TYPE);
                    break;

                case 'f':
                    expressionBuffer.append(REGEX_MEMBER);
                    break;

                case 'm':
                    expressionBuffer.append(REGEX_MEMBER);
                    break;

                case 'a':
                    expressionBuffer.append(REGEX_ARGUMENTS);
                    break;
            }

            expressionBuffer.append(')');

            expressionTypes[expressionTypeCount++] = expressionType;

            index = nextIndex + 2;
        }

        // Copy the last literal piece of the input line.
        expressionBuffer.append(regularExpression.substring(index));

        this.expressionTypeCount = expressionTypeCount;
        this.pattern             = Pattern.compile(expressionBuffer.toString());
        this.verbose             = verbose;
    }


    /**
     * Parses all frame information from a given line.
     * @param  line a line that represents a stack frame.
     * @return the parsed information, or null if the line doesn't match a
     *         stack frame.
     */
    public FrameInfo parse(String line)
    {
        // Try to match it against the regular expression.
        Matcher matcher = pattern.matcher(line);

        if (!matcher.matches())
        {
            return null;
        }

        // The line matched the regular expression.
        String className  = null;
        String sourceFile = null;
        int    lineNumber = 0;
        String type       = null;
        String fieldName  = null;
        String methodName = null;
        String arguments  = null;

        // Extract a class name, a line number, a type, and
        // arguments.
        for (int expressionTypeIndex = 0; expressionTypeIndex < expressionTypeCount; expressionTypeIndex++)
        {
            int startIndex = matcher.start(expressionTypeIndex + 1);
            if (startIndex >= 0)
            {
                String match = matcher.group(expressionTypeIndex + 1);

                char expressionType = expressionTypes[expressionTypeIndex];
                switch (expressionType)
                {
                    case 'c':
                        className = match;
                        break;

                    case 'C':
                        className = ClassUtil.externalClassName(match);
                        break;

                    case 's':
                        sourceFile = match;
                        break;

                    case 'l':
                        lineNumber = Integer.parseInt(match);
                        break;

                    case 't':
                        type = match;
                        break;

                    case 'f':
                        fieldName = match;
                        break;

                    case 'm':
                        methodName = match;
                        break;

                    case 'a':
                        arguments = match;
                        break;
                }
            }
        }

        return new FrameInfo(className,
                             sourceFile,
                             lineNumber,
                             type,
                             fieldName,
                             methodName,
                             arguments);
    }


    /**
     * Formats the given frame information based on the given template line.
     * It is the reverse of {@link #parse(String)}, but optionally with
     * different frame information.
     * @param  line      a template line that represents a stack frame.
     * @param  frameInfo information about a stack frame.
     * @return the formatted line, or null if the line doesn't match a
     *         stack frame.
     */
    public String format(String line, FrameInfo frameInfo)
    {
        // Try to match it against the regular expression.
        Matcher matcher = pattern.matcher(line);

        if (!matcher.matches())
        {
            return null;
        }

        StringBuffer formattedBuffer = new StringBuffer();

        int lineIndex = 0;
        for (int expressionTypeIndex = 0; expressionTypeIndex < expressionTypeCount; expressionTypeIndex++)
        {
            int startIndex = matcher.start(expressionTypeIndex + 1);
            if (startIndex >= 0)
            {
                int    endIndex = matcher.end(expressionTypeIndex + 1);
                String match    = matcher.group(expressionTypeIndex + 1);

                // Copy a literal piece of the input line.
                formattedBuffer.append(line.substring(lineIndex, startIndex));

                // Copy a matched and translated piece of the input line.
                char expressionType = expressionTypes[expressionTypeIndex];
                switch (expressionType)
                {
                    case 'c':
                        formattedBuffer.append(frameInfo.getClassName());
                        break;

                    case 'C':
                        formattedBuffer.append(ClassUtil.internalClassName(frameInfo.getClassName()));
                        break;

                    case 's':
                        formattedBuffer.append(frameInfo.getSourceFile());
                        break;

                    case 'l':
                        formattedBuffer.append(frameInfo.getLineNumber());
                        break;

                    case 't':
                        formattedBuffer.append(frameInfo.getType());
                        break;

                    case 'f':
                        if (verbose)
                        {
                            formattedBuffer.append(frameInfo.getType()).append(' ');
                        }
                        formattedBuffer.append(frameInfo.getFieldName());
                        break;

                    case 'm':
                        if (verbose)
                        {
                            formattedBuffer.append(frameInfo.getType()).append(' ');
                        }
                        formattedBuffer.append(frameInfo.getMethodName());
                        if (verbose)
                        {
                            formattedBuffer.append('(').append(frameInfo.getArguments()).append(')');
                        }
                        break;

                    case 'a':
                        formattedBuffer.append(frameInfo.getArguments());
                        break;
                }

                // Skip the original element whose replacement value
                // has just been appended.
                lineIndex = endIndex;
            }
        }

        // Copy the last literal piece of the input line.
        formattedBuffer.append(line.substring(lineIndex));

        // Return the formatted line.
        return formattedBuffer.toString();
    }
}
